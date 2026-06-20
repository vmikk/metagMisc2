#include <Rcpp.h>

#include <cmath>
#include <string>

namespace {

inline bool is_positive_finite(double x) {
  return std::isfinite(x) && x > 0.0;
}

}  // namespace

// [[Rcpp::export(name = ".sparse_to_long_table_cpp")]]
Rcpp::List sparse_to_long_table_cpp_impl(Rcpp::S4 mat) {
  if (!mat.hasSlot("x") || !mat.hasSlot("i") || !mat.hasSlot("p") ||
      !mat.hasSlot("Dim") || !mat.hasSlot("Dimnames")) {
    Rcpp::stop("mat must be a dgCMatrix-like object");
  }

  Rcpp::NumericVector x = mat.slot("x");
  Rcpp::IntegerVector i = mat.slot("i");
  Rcpp::IntegerVector p = mat.slot("p");
  Rcpp::IntegerVector dim = mat.slot("Dim");
  Rcpp::List dimnames = mat.slot("Dimnames");

  if (dim.size() != 2) {
    Rcpp::stop("mat@Dim must have length 2");
  }
  const int nrow = dim[0];
  const int ncol = dim[1];
  if (nrow < 0 || ncol < 0) {
    Rcpp::stop("matrix dimensions must be non-negative");
  }
  if (p.size() != ncol + 1) {
    Rcpp::stop("mat@p length does not match the number of columns");
  }
  if (i.size() != x.size()) {
    Rcpp::stop("mat@i and mat@x must have the same length");
  }

  Rcpp::CharacterVector otu_names;
  Rcpp::CharacterVector sample_names;
  bool has_otu_names = false;
  bool has_sample_names = false;

  if (dimnames.size() >= 2) {
    Rcpp::RObject otu_obj = dimnames[0];
    Rcpp::RObject sample_obj = dimnames[1];

    if (!Rf_isNull(otu_obj)) {
      otu_names = Rcpp::as<Rcpp::CharacterVector>(otu_obj);
      if (otu_names.size() != nrow) {
        Rcpp::stop("matrix row names length must match nrow");
      }
      has_otu_names = true;
    }
    if (!Rf_isNull(sample_obj)) {
      sample_names = Rcpp::as<Rcpp::CharacterVector>(sample_obj);
      if (sample_names.size() != ncol) {
        Rcpp::stop("matrix column names length must match ncol");
      }
      has_sample_names = true;
    }
  }

  R_xlen_t n_positive = 0;
  for (int col = 0; col < ncol; ++col) {
    const int p0 = p[col];
    const int p1 = p[col + 1];
    if (p0 < 0 || p1 < p0 || p1 > x.size()) {
      Rcpp::stop("mat@p contains invalid column pointers");
    }
    for (int t = p0; t < p1; ++t) {
      if (i[t] < 0 || i[t] >= nrow) {
        Rcpp::stop("mat@i contains row indices outside matrix dimensions");
      }
      if (is_positive_finite(x[t])) {
        ++n_positive;
      }
    }
  }

  Rcpp::CharacterVector sample_id(n_positive);
  Rcpp::CharacterVector otu(n_positive);
  Rcpp::NumericVector abundance(n_positive);

  R_xlen_t out = 0;
  for (int col = 0; col < ncol; ++col) {
    const std::string fallback_sample =
        has_sample_names ? std::string() : "Sample" + std::to_string(col + 1);

    for (int t = p[col]; t < p[col + 1]; ++t) {
      const double value = x[t];
      if (!is_positive_finite(value)) {
        continue;
      }

      if (has_sample_names) {
        sample_id[out] = sample_names[col];
      } else {
        sample_id[out] = fallback_sample;
      }

      if (has_otu_names) {
        otu[out] = otu_names[i[t]];
      } else {
        otu[out] = "OTU" + std::to_string(i[t] + 1);
      }

      abundance[out] = value;
      ++out;
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("SampleID") = sample_id,
      Rcpp::Named("OTU") = otu,
      Rcpp::Named("Abundance") = abundance);
}

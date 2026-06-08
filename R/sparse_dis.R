sparse_dis <- function(x,
  dissim           = "bray_curtis",
  phy_tree         = NULL,
  unifrac_alpha    = 0.5,
  n_threads        = 1L,
  samples_are_rows = FALSE) {

  all_beta <- c("bray_curtis", "bray_curtis_pa", "euclidean",
                "hellinger", "simpson", "simpson_pa")
  all_dissim <- c(all_beta, .unifrac_metric_names())

  bad <- setdiff(dissim, all_dissim)
  if (length(bad)) {
    stop("Unknown dissim metrics: ", paste(bad, collapse = ", "), call. = FALSE)
  }
  if (!length(dissim)) {
    stop("Select at least one dissim metric", call. = FALSE)
  }

  if (any(.is_unifrac_metric(dissim))) {
    if (!is.numeric(unifrac_alpha) || length(unifrac_alpha) != 1L ||
        is.na(unifrac_alpha) || unifrac_alpha < 0 || unifrac_alpha > 1) {
      stop("unifrac_alpha must be a numeric scalar in the 0-1 range",
           call. = FALSE)
    }
  }

  mat <- as_rarefy_matrix(x, samples_are_rows = samples_are_rows)
  if (!inherits(mat, "dgCMatrix")) {
    mat <- methods::as(mat, "dgCMatrix")
  }

  cn <- colnames(mat)
  if (is.null(cn)) {
    cn <- paste0("Sample", seq_len(ncol(mat)))
    colnames(mat) <- cn
  }

  beta_only    <- dissim[!.is_unifrac_metric(dissim)]
  unifrac_only <- dissim[.is_unifrac_metric(dissim)]

  phylo_tree <- NULL
  if (length(unifrac_only)) {
    phylo_tree <- .prepare_phylo_tree(
      phy_tree, rownames(mat),
      context = "UniFrac metrics")
  }

  beta_mats <- sparse_beta_cpp(
    mat,
    as.character(beta_only),
    as.character(unifrac_only),
    if (!is.null(phylo_tree)){ phylo_tree$tree } else { NULL },
    if (!is.null(phylo_tree)){ as.integer(phylo_tree$row_to_tip) } else { NULL },
    as.double(unifrac_alpha),
    as.integer(n_threads)
  )

  ## Reorder to match user-supplied dissim order and convert to dist
  out <- lapply(dissim, function(dm) {
    mfull <- beta_mats[[dm]]
    rownames(mfull) <- cn
    colnames(mfull) <- cn
    stats::as.dist(mfull, diag = FALSE, upper = FALSE)
  })
  names(out) <- dissim
  out
}

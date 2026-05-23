rarefy_multi <- function(x,
  depth     = NULL,
  n_iter    = 100L,
  metrics   = c("richness", "shannon", "hill1", "hill2"),
  kernel    = c("hypergeometric", "permutation"),
  samples_are_rows = FALSE) {
  kernel_code <- if (identical(kernel, "hypergeometric")) 0L else 1L
  mat <- as_rarefy_matrix(x, samples_are_rows = samples_are_rows)
  cn <- colnames(mat)
  cs <- Matrix::colSums(mat)
  cn <- colnames(mat)
  depths <- as.integer(depth)
  allm   <- .alpha_metric_names()
  countm <- .count_alpha_metric_names()
  res <- rarefy_alpha_cpp(
    mat,
    depths,
    as.integer(n_iter),
    as.integer(kernel_code)
  )

## Supported alpha diversity metrics
.alpha_metric_names <- function() {
  c(.count_alpha_metric_names(), .phylo_alpha_metric_names())
}

.count_alpha_metric_names <- function() {
  c("richness", "shannon", "hill1", "hill2", "simpson_dom", "evenness")
}

.phylo_alpha_metric_names <- function() {
  c("faith_pd")
}
}

rarefy_multi <- function(x,
  depth     = NULL,
  n_iter    = 100L,
  metrics   = c("richness", "shannon", "hill1", "hill2"),
  kernel    = c("hypergeometric", "permutation"),
  samples_are_rows = FALSE) {
  kernel_code <- if (identical(kernel, "hypergeometric")) 0L else 1L
  mat <- as_rarefy_matrix(x, samples_are_rows = samples_are_rows)
  if (!inherits(mat, "dgCMatrix")) {
    mat <- methods::as(mat, "dgCMatrix")
  }

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

.alpha_list_to_df <- function(res, sample_names, depths, metrics) {
  ns    <- length(sample_names)
  nd    <- length(depths)
  total <- ns * nd * length(metrics)
  sample_col <- character(total)
  depth_col  <- integer(total)
  metric_col <- character(total)
  mean_col   <- numeric(total)
  sd_col     <- numeric(total)
  min_col    <- numeric(total)
  max_col    <- numeric(total)
  ri <- 1L
  for (m in metrics) {
    mm <- res$mean[[m]]
    ss <- res$sd[[m]]
    mi <- res$min[[m]]
    ma <- res$max[[m]]
    for (si in seq_len(ns)) {
      for (di in seq_len(nd)) {
        sample_col[ri] <- sample_names[si]
        depth_col[ri]  <- depths[di]
        metric_col[ri] <- m
        mean_col[ri]   <- mm[si, di]
        sd_col[ri]     <- ss[si, di]
        min_col[ri]    <- mi[si, di]
        max_col[ri]    <- ma[si, di]
        ri <- ri + 1L
      }
    }
  }
  res <- data.frame(
    sample = sample_col,
    depth  = depth_col,
    metric = metric_col,
    mean   = mean_col,
    sd     = sd_col,
    min    = min_col,
    max    = max_col,
    stringsAsFactors = FALSE)
  return(res)
}
}

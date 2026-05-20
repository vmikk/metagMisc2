sparse_div <- function(x,
  metrics          = c("richness", "shannon", "hill1", "hill2"),
  phy_tree         = NULL,
  n_threads        = 1L,
  samples_are_rows = FALSE) {

  mat <- as_rarefy_matrix(x, samples_are_rows = samples_are_rows)
  if (!inherits(mat, "dgCMatrix")) {
    mat <- methods::as(mat, "dgCMatrix")
  }

  cn <- colnames(mat)
  if (is.null(cn)) {
    cn <- paste0("Sample", seq_len(ncol(mat)))
    colnames(mat) <- cn
  }

  allm   <- .alpha_metric_names()
  countm <- .count_alpha_metric_names()
  phylom <- .phylo_alpha_metric_names()

  bad <- setdiff(metrics, allm)
  if (length(bad)) {
    stop("Unknown metrics: ", paste(bad, collapse = ", "), call. = FALSE)
  }
  if (!length(metrics)) {
    stop("Select at least one metric", call. = FALSE)
  }

  count_mask <- rep(FALSE, length(countm))
  names(count_mask) <- countm
  count_mask[intersect(metrics, countm)] <- TRUE

  phylo_mask <- rep(FALSE, length(phylom))
  names(phylo_mask) <- phylom
  phylo_mask[intersect(metrics, phylom)] <- TRUE

  phylo_tree <- NULL
  if (any(phylo_mask)) {
    phylo_tree <- .prepare_phylo_tree(
      phy_tree,
      rownames(mat),
      context = "faith_pd")
  }

  res <- sparse_alpha_cpp(
    mat,
    as.logical(count_mask),
    as.logical(phylo_mask),
    if (!is.null(phylo_tree)){ phylo_tree$tree } else { NULL },
    if (!is.null(phylo_tree)){ as.integer(phylo_tree$row_to_tip) } else { NULL },
    as.integer(n_threads) )

  ## Flatten to long data.frame
  ns <- length(cn)
  req_count  <- names(count_mask)[count_mask]
  req_phylo  <- names(phylo_mask)[phylo_mask]
  req_all    <- intersect(metrics, c(req_count, req_phylo))  # preserve order
  total_rows <- ns * length(req_all)

  sample_col <- character(total_rows)
  metric_col <- character(total_rows)
  value_col  <- numeric(total_rows)

  ri <- 1L
  for (m in req_all) {
    if (m %in% countm) {
      vals <- res$count[[m]]
    } else {
      vals <- res$phylo[[m]]
    }
    for (si in seq_len(ns)) {
      sample_col[ri] <- cn[si]
      metric_col[ri] <- m
      value_col[ri]  <- vals[si]
      ri <- ri + 1L
    }
  }

  res <- data.frame(
    sample = sample_col,
    metric = metric_col,
    value  = value_col,
    stringsAsFactors = FALSE)

  return(res)
}

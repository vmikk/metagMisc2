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
  phylom <- .phylo_alpha_metric_names()
  mch <- match(metrics, allm)
  if (anyNA(mch)) {
    stop("Unknown metrics: ", paste(setdiff(metrics, allm), collapse = ", "))
  }
  count_mask <- rep(FALSE, length(countm))
  names(count_mask) <- countm
  count_mask[intersect(metrics, countm)] <- TRUE
  phylo_mask <- rep(FALSE, length(phylom))
  names(phylo_mask) <- phylom
  phylo_mask[intersect(metrics, phylom)] <- TRUE

  ## Prepare the phylo tree once for both phylogenetic 
  ## alpha metrics (Faith PD) and UniFrac dissimilarities
  need_tree <- any(phylo_mask) ||
               (!is.null(dissim) && any(.is_unifrac_metric(dissim)))
  phylo_tree <- NULL
  if (need_tree) {
    phylo_tree <- .prepare_phylo_tree(
      phy_tree,
      rownames(mat),
      context = "phylogenetic metrics")
  }

  ## Alpha kernel: count metrics + Faith PD in a single rarefaction pass
  res <- rarefy_alpha_cpp(
    mat,
    depths,
    as.integer(n_iter),
    as.logical(count_mask),
    as.logical(phylo_mask),
    if (!is.null(phylo_tree)){ phylo_tree$tree } else { NULL },
    if (!is.null(phylo_tree)){ as.integer(phylo_tree$row_to_tip) } else { NULL },
    as.integer(n_threads),
    as.double(seed),
    as.integer(kernel_code)
  )
  alpha_parts <- list()
  if (any(count_mask)) {
    alpha_parts[[length(alpha_parts) + 1L]] <-
      .alpha_list_to_df(
        res$count,
        sample_names = cn,
        depths = depths,
        metrics = names(count_mask)[count_mask])
  }
  if (any(phylo_mask)) {
    alpha_parts[[length(alpha_parts) + 1L]] <-
      .alpha_list_to_df(
        res$phylo,
        sample_names = cn,
        depths = depths,
        metrics = names(phylo_mask)[phylo_mask])
  }
  alpha_df <- do.call(rbind, alpha_parts)

  beta <- NULL
  if (!is.null(dissim)) {
    d0           <- depths[1L]
    beta_only    <- dissim[!.is_unifrac_metric(dissim)]
    unifrac_only <- dissim[.is_unifrac_metric(dissim)]

    ## Beta kernel: rarefy once per rep;
    ## compute all metrics in one parallel pair loop
    beta_mats <- rarefy_beta_cpp(
      mat,
      d0,
      as.integer(n_iter),
      as.character(beta_only),
      as.character(unifrac_only),
      if (!is.null(phylo_tree)) phylo_tree$tree else NULL,
      if (!is.null(phylo_tree)) as.integer(phylo_tree$row_to_tip) else NULL,
      as.double(unifrac_alpha),
      as.integer(n_threads),
      as.double(seed),
      as.integer(kernel_code)
    )
    ## Reorder output to match the original dissim order and attach names
    beta <- lapply(dissim, function(dm) {
      mfull <- beta_mats[[dm]]
      rownames(mfull) <- cn
      colnames(mfull) <- cn
      stats::as.dist(mfull, diag = FALSE, upper = FALSE)
    })
    names(beta) <- dissim
  }
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

.unifrac_metric_names <- function() {
  c("unifrac_unweighted", "unifrac_weighted", "unifrac_generalized", "unifrac_vaw")
}

.is_unifrac_metric <- function(x) {
  x %in% .unifrac_metric_names()
}

.prepare_phylo_tree <- function(phy_tree, taxa, context) {
  if (is.null(phy_tree)) {
    stop("phy_tree is required when requesting ", context, call. = FALSE)
  }
  if (!inherits(phy_tree, "phylo")) {
    stop("phy_tree must inherit from class 'phylo'", call. = FALSE)
  }
  if (is.null(taxa) || anyNA(taxa) || any(!nzchar(taxa))) {
    stop("matrix row names are required to match taxa to phy_tree tip labels", call. = FALSE)
  }
  if (anyDuplicated(taxa)) {
    stop("matrix row names must be unique for phylogenetic tree metrics", call. = FALSE)
  }
  tips <- phy_tree$tip.label
  if (is.null(tips) || anyNA(tips) || any(!nzchar(tips))) {
    stop("phy_tree must contain non-missing tip labels", call. = FALSE)
  }
  if (anyDuplicated(tips)) {
    stop("phy_tree tip labels must be unique", call. = FALSE)
  }
  missing_tips <- setdiff(taxa, tips)
  if (length(missing_tips)) {
    stop(
      "phy_tree is missing taxa present in the matrix: ",
      paste(utils::head(missing_tips, 10L), collapse = ", "),
      if (length(missing_tips) > 10L) ", ..." else "",
      call. = FALSE)
  }
  extra_tips <- setdiff(tips, taxa)
  if (length(extra_tips)) {
    phy_tree <- ape::drop.tip(phy_tree, extra_tips)
  }
  if (is.null(phy_tree$edge) || !is.matrix(phy_tree$edge) || ncol(phy_tree$edge) != 2L) {
    stop("phy_tree$edge must be a two-column matrix", call. = FALSE)
  }
  if (is.null(phy_tree$edge.length)) {
    stop("phy_tree$edge.length is required for UniFrac", call. = FALSE)
  }
  if (length(phy_tree$edge.length) != nrow(phy_tree$edge)) {
    stop("phy_tree$edge.length must have one value per edge", call. = FALSE)
  }
  if (anyNA(phy_tree$edge.length) || any(phy_tree$edge.length < 0)) {
    stop("phy_tree$edge.length must contain non-negative, non-missing values", call. = FALSE)
  }
  phy_tree <- ape::reorder.phylo(phy_tree, order = "postorder")
  row_to_tip <- match(taxa, phy_tree$tip.label) - 1L
  if (anyNA(row_to_tip)) {
    stop("failed to align matrix taxa with phy_tree tip labels", call. = FALSE)
  }
  list(tree = phy_tree, row_to_tip = row_to_tip)
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

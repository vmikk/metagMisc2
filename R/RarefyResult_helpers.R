print.RarefyResult <- function(x, ...) {
  cat("RarefyResult: ", NROW(x$alpha), " alpha rows\n", sep = "")
  if (!is.null(x$beta)) {
    cat("  beta: ", paste(names(x$beta), collapse = ", "), "\n", sep = "")
  }
  if (!is.null(x$tables)) {
    cat("  tables: ", length(x$tables), " sparse matrix(ices)\n", sep = "")
  }
  if (length(x$dropped_samples)) {
    cat("  dropped samples: ", length(x$dropped_samples), "\n", sep = "")
  }
  invisible(x)
}

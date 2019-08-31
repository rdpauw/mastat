listfeats <- function(x){
  
  
  features <- x
  features <- split(features, rep(1:ncol(features), each = nrow(features)))
  features <- lapply(features, function(x) x[!is.na(x)])
  
  return(features)
  
}

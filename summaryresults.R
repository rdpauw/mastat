summaryresults <- function(x, name){
  
  tmp <- x
  tmp$accuracy <- as.numeric(tmp$accuracy)
  tmp$sens <- as.numeric(tmp$sens)
  tmp$spec <- as.numeric(tmp$spec)
  tmp$precision <- as.numeric(tmp$tp) / 
    (as.numeric(tmp$tp) + as.numeric(tmp$tn))
  tmp$auc <- 1/2 * (tmp$sens + tmp$spec)
  
  
  tab_acc <- tmp[,c("type","accuracy", "sens", "spec", "precision", "auc")] %>%
    group_by(type) %>%
    summarise_if(is.numeric, list(~mean(., na.rm = TRUE), ~sd(., na.rm = TRUE), ~min(., na.rm = TRUE), ~max(., na.rm = TRUE)), na.rm = TRUE)
  
  write.csv2(tab_acc,paste0("c://Users/rdpauw/OneDrive - UGent/MSc in Statistical Data Analysis/Thesis/Nekpijn/",name))
  
}
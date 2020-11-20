convert_sql_to_dplyr <- function(sql_query) {
  x <- sql_query %>% tolower()
  
  patterns = c(" in ", " and ", " or ", "oscd.|dod.|doad.|DOD.", " = ", 'true', 'false') #these are alias for sql tables
  replacement = c(" %in% list", " & ", " | ", "", " == ", 'TRUE', 'FALSE')
  
  midpoint_query <- stringr::str_replace_all(x, setNames(replacement, patterns))
  
  R_query <- midpoint_query %>% 
    gsub('(\\w+) is null', 'is.na(\\1)', .) %>% 
    gsub("(\\w+) ilike '(\\w+)'", "str_detect(\\1, '\\2')", .)
    
  return(R_query)
}

-- Approach
-- Split the table into two : page_load and page_exit
-- Remove dulicates
-- Join tables
-- average session time per user

with page_loads as
(select user_id, date(timestamp) as date, max(timestamp) as load_time, action 
from facebook_web_log
where action = "page_load"
group by user_id, date),

page_exits as
(select user_id, date(timestamp) as date, min(timestamp) as exit_time, action 
from facebook_web_log
where action = "page_exit"
group by 1, 2)

select pl.user_id, avg(timestampdiff(second, load_time, exit_time))
from page_loads pl
join page_exits pe
on pl.user_id = pe.user_id and pl.date = pe.date
group by 1;
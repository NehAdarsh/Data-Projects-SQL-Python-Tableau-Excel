## Question: Users By Average Session Time (ID 10352)
## Question Link: https://platform.stratascratch.com/coding/10352-users-by-avg-session-time?code_type=3
## Asked by Meta/Facebook
## Difficulty: Medium

### Description: Calculate each user's average session time. A session is defined as the time difference between a page_load and page_exit. For simplicity, assume a user has only 1 session per day and if there are multiple of the same events on that day, consider only the latest page_load and earliest page_exit, with an obvious restriction that the load time event should happen before the exit time event. Output the user_id and their average session time.

My Approach to tackle this problem:
There are 2 columns page_load and page_exits that we need to take care of. So, I thought to divide these two columns into two separate tables, just for simplicity. 
### Step 1: Split the table into two: page_load and page_exit

`select * 
from facebook_web_log
where action = "page_load" `

`select * 
from facebook_web_log
where action = "page_exit" `

### Step 2: Now that we have our two separate tables for page_load and page_exit.
- Next, we will extract the date from the timestamp column
- As we need to consider only the latest page_load and earliest page_exit, we will use MAX and MIN with the timestamp
- In the end, we group it by user_id and the date
  
`select user_id, date(timestamp) as date, max(timestamp) as load_time, action 
from facebook_web_log
where action = "page_load"
group by user_id, date`

`select user_id, date(timestamp) as date, min(timestamp) as exit_time, action 
from facebook_web_log
where action = "page_exit"
group by 1, 2`

### Step 3: Create CTEs for these two tables and join them
  
`with page_loads as
(select user_id, date(timestamp) as date, max(timestamp) as load_time, action 
from facebook_web_log
where action = "page_load"
group by user_id, date),`

`page_exits as
(select user_id, date(timestamp) as date, min(timestamp) as exit_time, action 
from facebook_web_log
where action = "page_exit"
group by 1, 2)`

`Select * 
from page_loads pl
join page_exits pe
on pl.user_id = pe.user_id and pl.date = pe.date`

### Step 4: Extract the user_id and avg timestamp diff between load_time and exit_time
  
`select pl.user_id, avg(timestampdiff(second, load_time, exit_time))
from page_loads pl
join page_exits pe
on pl.user_id = pe.user_id and pl.date = pe.date
group by 1; `

-------------------------------------------------------------------------------------------

### Full Query:
`with page_loads as
(select user_id, date(timestamp) as date, max(timestamp) as load_time, action 
from facebook_web_log
where action = "page_load"
group by user_id, date),`

`page_exits as
(select user_id, date(timestamp) as date, min(timestamp) as exit_time, action 
from facebook_web_log
where action = "page_exit"
group by 1, 2)`

`select pl.user_id, avg(timestampdiff(second, load_time, exit_time))
from page_loads pl
join page_exits pe
on pl.user_id = pe.user_id and pl.date = pe.date
group by 1;`



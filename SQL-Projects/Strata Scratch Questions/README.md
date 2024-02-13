## Question: Users By Average Session Time (ID 10352)
## Question Link: https://platform.stratascratch.com/coding/10352-users-by-avg-session-time?code_type=3
## Asked by Meta/Facebook
## Difficulty: Medium

### Description: Calculate each user's average session time. A session is defined as the time difference between a page_load and page_exit. For simplicity, assume a user has only 1 session per day and if there are multiple of the same events on that day, consider only the latest page_load and earliest page_exit, with an obvious restriction that load time event should happen before exit time event . Output the user_id and their average session time.

### 
user_id	timestamp	action
0	4/25/19 13:30	page_load
0	4/25/19 13:30	page_load
0	4/25/19 13:30	scroll_down
0	4/25/19 13:30	scroll_up
0	4/25/19 13:31	scroll_down
0	4/25/19 13:31	scroll_down
0	4/25/19 13:31	page_exit
1	4/25/19 13:40	page_load
1	4/25/19 13:40	scroll_down
1	4/25/19 13:40	scroll_down
1	4/25/19 13:40	scroll_down
1	4/25/19 13:40	scroll_down
1	4/25/19 13:40	scroll_down
1	4/25/19 13:40	page_exit
2	4/25/19 13:41	page_load
2	4/25/19 13:41	scroll_down
2	4/25/19 13:41	scroll_down
2	4/25/19 13:41	scroll_up
1	4/26/19 11:15	page_load
1	4/26/19 11:15	scroll_down
1	4/26/19 11:15	scroll_down
1	4/26/19 11:15	scroll_up
1	4/26/19 11:15	page_exit
0	4/28/19 14:30	page_load
0	4/28/19 14:30	page_load
0	4/28/19 13:30	scroll_down
0	4/28/19 15:31	page_exit![image](https://github.com/NehAdarsh/Data-Projects-SQL-Python-Tableau-Excel/assets/111151093/3c59ad39-64c6-4bec-a5f3-6d07a8292ff2)

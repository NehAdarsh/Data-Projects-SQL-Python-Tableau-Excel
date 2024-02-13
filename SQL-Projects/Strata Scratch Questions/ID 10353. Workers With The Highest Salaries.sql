/* Question: Workers With The Highest Salaries

Description: You have been asked to find the job titles of the highest-paid employees.
Your output should include the highest-paid title or multiple titles with the same salary.

Asked by Amazon and DoorDash
Difficulty level: Medium
Question ID 10353
Link: https://platform.stratascratch.com/coding/10353-workers-with-the-highest-salaries?code_type=1
*/

select worker_title 
from worker as w
join title as t
on w.worker_id = t.worker_ref_id
order by salary desc
limit 2;
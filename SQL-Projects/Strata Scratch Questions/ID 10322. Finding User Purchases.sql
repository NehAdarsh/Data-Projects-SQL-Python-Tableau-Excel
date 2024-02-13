-- Self Join tables
-- Using DATEDIFF in where condition, compare transaction dates

select a1.user_id
from amazon_transactions a1
join amazon_transactions a2
on a1.user_id = a2.user_id 
and a1.id <> a2.id 
where datediff(a2.created_at, a1.created_at) between 0 and 7 
group by 1;
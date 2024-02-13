select company, sum(profits) as total_profits
from forbes_global_2010_2014
group by 1
order by 2 desc
limit 3;
-- EDA (Exploratory Data Analysis) --

USE SalesDataWalmart;

-- --------- 1. Basic Questions -------------

-- Q1: How many unique cities does the data have?
SELECT distinct city
From sales;

-- Q2: In which city is each branch?
Select city, branch
from Sales
group by 1, 2;

-- --------- 2. Product Analysis -- ---------
-- Q1. How many unique product lines does the data have?
Select Count(Distinct product_line)
From sales;

-- Q2. What is the most common payment method?
Select payment, count(payment)
from sales
group by 1
order by 2 desc;

-- Q3. What is the most selling product line?
Select product_line, count(product_line)
From sales
group by 1
order by 2 desc
limit 1;

-- Q4. What is the total revenue by month?
Select Month_name, SUM(total) as revenue
from sales
group by 1
order by 2 desc;

-- Q5. What month had the largest COGS?
Select month_name, sum(cogs)
From Sales
group by 1
order by 2 desc;

-- Q6. What product line had the largest revenue?
Select product_line, sum(total)
from sales
group by 1
order by 2 desc;

-- Q7. What is the city with the largest revenue?
Select city, sum(total)
from sales
group by 1
order by 2 desc;

-- Q8. What product line had the largest tax?
Select product_line, sum(tax)
from sales
group by 1
order by 2 desc;

-- Q10. Which branch sold more products than average product sold?
select branch, sum(quantity)
from sales
group by 1
having sum(quantity) > (select avg(quantity) from sales);

-- Q11. What is the most common product line by gender?
Select product_line, count(gender)
from sales
group by 1;

-- Q12. What is the average rating of each product line?
Select product_line, round(Avg(rating), 2)
from sales
group by 1;

-- --------- 3. Sales Analysis -- ---------

-- Q1. Number of sales made in each time of the day per weekday
select time_of_day, count(*) as sales
from sales
group by 1;

-- Q2. Which of the customer types brings the most revenue?
Select customer_type, sum(total)
from Sales
group by 1
order by 2 desc;

-- Q3. Which city has the largest tax percent/ VAT (**Value Added Tax**)?
select city, round(avg(tax),2)
from sales
group by 1
order by 2 desc;

-- Q4. Which customer type pays the most in VAT?
select customer_type, sum(tax)
from sales
group by 1
order by 2 desc;

-- --------- 4. Customer Analysis -- ---------

-- Q1. How many unique customer types does the data have?
select distinct customer_type
from sales;

-- Q2. How many unique payment methods does the data have?
select distinct payment
from Sales;

-- Q3. What is the most common customer type?
select customer_type, count(*) as cnt
from Sales
group by 1
order by 2;

-- Q4. Which customer type buys the most?
select customer_type, count(*)
from sales
group by 1
order by 2 desc;

-- Q5. What is the gender of most of the customers?
select gender, count(gender)
from sales
group by 1;

-- Q6. What is the gender distribution per branch?
select branch, count(gender)
from sales
group by 1
order by 2 desc;

-- Q7. Which time of the day do customers give most ratings?
select time_of_day, count(rating)
from sales
group by 1
order by 2 desc;

-- Q8. Which time of the day do customers give most ratings per branch?
select branch, time_of_day, count(rating)
from sales
group by 1, 2
order by 3 desc;

-- Q9. Which day fo the week has the best avg ratings?
select day_name, avg(rating)
from sales
group by 1
order by 2 desc;

-- Q10. Which day of the week has the best average ratings per branch?
select branch, avg(rating)
from sales
group by 1
order by 2 desc;

-- -------------<>--------------- --


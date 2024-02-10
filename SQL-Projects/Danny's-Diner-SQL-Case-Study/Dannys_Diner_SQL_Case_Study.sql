/* --------------------
   Case Study Questions
   --------------------*/
   
USE dannys_diner;


-- 1. What is the total amount each customer spent at the restaurant?

SELECT 
	t1.customer_id as customer, SUM(t2.price) as total_amt_spent
FROM 
	dannys_diner.sales as t1
JOIN 
	dannys_diner.menu as t2
ON 
	t1.product_id = t2.product_id
GROUP BY 1;

--- ---------------------------------------------------------------------
--- ---------------------------------------------------------------------

-- 2. How many days has each customer visited the restaurant?

SELECT
	t1.customer_id as customer, COUNT(DISTINCT t1.order_date) as n_of_days
FROM 
	dannys_diner.sales as t1
GROUP BY 1;

--- ---------------------------------------------------------------------
--- ---------------------------------------------------------------------
    
-- 3. What was the first item from the menu purchased by each customer?

WITH cte as (
  SELECT 
  	t1.customer_id, t1.order_date, t2.product_name, 
  	DENSE_RANK() OVER (PARTITION BY t1.customer_id ORDER BY t1.order_date) as Rank_
  FROM 
  	dannys_diner.sales as t1
  JOIN 
  	dannys_diner.menu as t2
  ON 
  	t1.product_id = t2.product_id)
    
--
  
SELECT 
	customer_id as customer, product_name
FROM 
	cte
WHERE 
	rank_ = 1
GROUP BY 1, 2;


--- ---------------------------------------------------------------------
--- ---------------------------------------------------------------------

-- 4. What is the most purchased item on the menu and how many times was it purchased by all customers?

SELECT 
	t2.product_name as product, count(t1.product_id) as n_of_time_item_purchased
FROM 
	dannys_diner.sales as t1
JOIN 
	dannys_diner.menu as t2
ON 
	t1.product_id = t2.product_id
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1;

--- ---------------------------------------------------------------------
--- ---------------------------------------------------------------------

-- 5. Which item was the most popular for each customer?
SELECT 
	t2.product_name as product, COUNT(t1.product_id) as total_count
FROM 
	dannys_diner.sales as t1
JOIN 
	dannys_diner.menu as t2
ON 
	t1.product_id = t2.product_id
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1;

--- ---------------------------------------------------------------------
--- ---------------------------------------------------------------------

-- 6. Which item was purchased first by the customer after they became a member?

WITH CTE_ as(
  
SELECT 
	t1.customer_id as customer, t2.product_name as product, join_date, order_date,
  DENSE_RANK() OVER (PARTITION BY t1.customer_id ORDER BY order_date) as rank_
FROM 
	dannys_diner.sales as t1
JOIN 
	dannys_diner.menu as t2
ON 
	t1.product_id = t2.product_id
JOIN 
	dannys_diner.members as t3
ON
	t3.customer_id = t1.customer_id
WHERE 
	join_date <= order_date
)

SELECT customer, product
from CTE_
WHERE rank_ = 1;

--- ---------------------------------------------------------------------
--- ---------------------------------------------------------------------

-- 7. Which item was purchased just before the customer became a member?
WITH CTE__ as(
  
SELECT 
	t1.customer_id as customer, t2.product_name as product, t2.product_id, join_date, order_date,
  DENSE_RANK() OVER (PARTITION BY t1.customer_id ORDER BY order_date DESC) as rank_
FROM 
	dannys_diner.sales as t1
JOIN 
	dannys_diner.menu as t2
ON 
	t1.product_id = t2.product_id
JOIN 
	dannys_diner.members as t3
ON
	t3.customer_id = t1.customer_id
WHERE 
	join_date > order_date
)


SELECT customer, product
FROM CTE__
WHERE rank_ = 1;


--- ---------------------------------------------------------------------
--- ---------------------------------------------------------------------

-- 8. What is the total items and amount spent for each member before they became a member?

SELECT 
	t1.customer_Id as customer, 
	COUNT(DISTINCT t1.product_id) as total_items_count, 
	SUM(t2.price) as total_spent
FROM dannys_diner.sales as t1
JOIN dannys_diner.menu as t2
ON t1.product_id = t2.product_id
JOIN dannys_diner.members as t3
ON t1.customer_id = t3.customer_id
WHERE order_date < join_date
GROUP BY t1.customer_Id;
--- ---------------------------------------------------------------------
--- ---------------------------------------------------------------------

-- 9. If each $1 spent equates to 10 points and sushi has a 2x points multiplier - how many points would each customer have?

WITH cte3 as (
SELECT product_id, product_name, price as actual_price, 
    CASE WHEN t2.product_name = 'sushi' THEN t2.price*20
    ELSE t2.price*10 END as total_points
FROM dannys_diner.menu as t2)


SELECT t1.customer_id as customer, sum(total_points) as Total_points_per_customer
FROM dannys_diner.sales as t1
JOIN cte3
ON t1.product_id = cte3.product_id
GROUP BY 1;

----------------------------------------------
--- ---------------------------------------------------------------------

-- 10. In the first week after a customer joins the program (including their join date) they earn 2x points on all items, not just sushi - how many points do customer A and B have at the end of January?

WITH dates AS 
(
   SELECT *, 
      DATE_ADD(join_date, INTERVAL 6 DAY) AS valid_date, 
      LAST_DAY('2021-01-31') AS last_date
   FROM members 
)

Select t1.Customer_id, 
	SUM( Case 
    When t2.product_ID = 1 THEN t2.price*20
	When t1.order_date between D.join_date and D.valid_date Then t2.price*20
	Else t2.price*10
	END ) as Points
From Dates D
join Sales t1
On D.customer_id = t1.customer_id
Join Menu t2
On t2.product_id = t1.product_id
Where t1.order_date < d.last_date
Group by t1.customer_id;









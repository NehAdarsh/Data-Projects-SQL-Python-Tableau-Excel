-- Feature Engineeering --


/* Let's add a few new columns that would be important to get insights into the sames data.*/

/* New Column 1: 'time_of_day` to give insight of sales in the Morning, Afternoon and Evening. 
This will help answer the question on which part of the day most sales are made.*/

USE SalesDataWalmart;
SELECT time, 
CASE 
	WHEN time between "00:00:00" and "12:00:00" THEN "Morning"
	WHEN time between"12:00:00" and "16:00:00" THEN "Afternoon"
	Else "Evening" 
END AS time_of_day
FROM sales;

ALTER TABLE sales
ADD time_of_day varchar(50);

UPDATE Sales
SET time_of_day = (
CASE WHEN time BETWEEN '00:00:00' AND '12:00:00' THEN 'Morning' 
WHEN time BETWEEN '12:00:00' AND '16:00:00' THEN 'Afternoon' 
ELSE 'Evening' END)
WHERE invoice_ID IS NOT NULL;

/* New column 2: `day_name` that contains the extracted days of the week on which the given transaction took place (Mon, Tue, Wed, Thur, Fri).
This will help answer the question on which week of the day each branch is busiest. */
SELECT date, dayname(date) as day_name
FROM sales;

ALTER TABLE Sales
Add day_name varchar(20);

Update sales
Set day_name = DAYNAME(date); 

Select * from sales;

/* New Column 3: `month_name` that contains the extracted months of the year on which the given transaction took place (Jan, Feb, Mar). 
Help determine which month of the year has the most sales and profit.*/

SELECT date, monthname(date) as month_name
From sales;

ALTER TABLE Sales
Add month_name varchar(20);

Update sales
Set month_name = MONTHNAME(date); 





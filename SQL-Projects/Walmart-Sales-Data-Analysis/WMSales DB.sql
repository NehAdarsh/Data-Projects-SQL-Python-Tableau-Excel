CREATE DATABASE IF NOT EXISTS SalesDataWalmart;

USE SalesDataWalmart;

CREATE TABLE IF NOT EXISTS sales(
Invoice_ID varchar(50) Not Null Primary key,	
Branch varchar(5) Not Null,
City varchar (30) not null,	
Customer_type varchar (30) not null,	
Gender varchar(10) not null,	
Product_line varchar (100) not null,	
Unit_price decimal not null,	
Quantity int not null,	
Tax float not null,	
Total decimal not null,	
Date datetime not null,	
Time TIME not null,	
Payment varchar (15)  not null,
cogs decimal not null,	
gross_margin_percentage	float,
gross_income decimal not null,
Rating float
);

SELECT count(*) from sales;
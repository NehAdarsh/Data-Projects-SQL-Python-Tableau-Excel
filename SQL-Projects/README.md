# Everything about SQL
![image](https://github.com/NehAdarsh/Data-Projects-SQL-Python-Tableau-Excel/assets/111151093/3c171aaa-de28-4f12-b22a-c2a697edc498)

# DDL Commands

**CREATE DATABASE: Creates a new database.**   
```CREATE DATABASE DatabaseName;```


**DROP DATABASE: Deletes a database.**  
```DROP DATABASE DatabaseName;```


**CREATE TABLE: Creates a new table in a database.**   
```CREATE TABLE table_name(column1 datatype, column2 datatype, column2 datatype, PRIMARY KEY( one or more columns )```

**ALTER TABLE: Alters the structure of an existing table.**   
```ALTER TABLE table_name ADD column_name datatype;```	

**DROP TABLE: Removes a table from a database.**   
```DROP TABLE table_name;	```

**CREATE INDEX: Creates an index on a table to improve a specific query performance.**    
```CREATE UNIQUE INDEX indexname ON tablename (columnname1, columnname2, …);```

**CREATE VIEW: Creates a view, a virtual table based on one or more existing tables.**   
```CREATE VIEW view_name AS SELECT column1, column2, ... FROM table_name WHERE condition;```

**CREATE PROCEDURE: Creates a stored procedure, a precompiled SQL statement that can be run multiple times with different parameters.**   
```CREATE PROCEDURE sp_name(parameter_list) BEGIN statements; END;```

**CREATE FUNCTION: Creates a custom user-defined function that can be utilized in SQL statements.**   
```CREATE FUNCTION function_name [ (parameter datatype [, parameter datatype]) ] RETURNS return_datatype BEGIN declaration_section executable_section END;```


**CREATE TRIGGER: Creates a trigger, a type of stored procedure that is automatically executed when certain events occur, such as inserting, updating, or deleting data in a table.**   
```CREATE TRIGGER trigger_name {BEFORE | AFTER} {INSERT | UPDATE | DELETE} ON table_name FOR EACH ROW BEGIN -- Trigger body (SQL statements) END;```



```



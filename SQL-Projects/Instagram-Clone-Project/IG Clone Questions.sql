USE ig_clone;

-- All tables check
SELECT * FROM users;
SELECT * FROM photos;
SELECT * FROM comments;
SELECT * FROM likes;
SELECT * FROM follows;
SELECT * FROM tags;
SELECT * FROM photo_tags;

-- Questions
-- We want to reward our users who have been around the longest.  Find the 5 oldest users.
SELECT * FROM users
ORDER BY created_at DESC
LIMIT 5;

-- What day of the week do most users register on? We need to figure out when to schedule an ad campgain
SELECT DATE_FORMAT(created_at, "%W") as weekday, Count(*) as register_cnt
From users
Group by 1
Order by 2 desc;

-- We want to target our inactive users with an email campaign. Find the users who have never posted a photo
SELECT username
FROM users as u
Left Join photos as p
On u.id = p.user_id
Where p.id IS NULL;

-- We're running a new contest to see who can get the most likes on a single photo. WHO WON??!!
SELECT 
    username,
    photos.id,
    photos.image_url, 
    COUNT(*) AS total
FROM photos
INNER JOIN likes
    ON likes.photo_id = photos.id
INNER JOIN users
    ON photos.user_id = users.id
GROUP BY photos.id
ORDER BY total DESC
LIMIT 1;

-- Our Investors want to know how many times does the average user post?
-- total number of photos/total number of users
SELECT ROUND((SELECT COUNT(*)FROM photos)/(SELECT COUNT(*) FROM users),2);

-- user ranking by postings
SELECT users.username,COUNT(photos.image_url)
FROM users
JOIN photos ON users.id = photos.user_id
GROUP BY users.id
ORDER BY 2 DESC;

-- total numbers of users who have posted at least one time 
SELECT COUNT(DISTINCT(users.id)) AS total_number_of_users_with_posts
FROM users
JOIN photos ON users.id = photos.user_id;

-- A brand wants to know which hashtags to use in a post
-- What are the top 5 most commonly used hashtags?
SELECT tag_name, COUNT(tag_name) AS total
FROM tags
JOIN photo_tags ON tags.id = photo_tags.tag_id
GROUP BY tags.id
ORDER BY total DESC;


-- We have a small problem with bots on our site...
-- Find users who have liked every single photo on the site
SELECT users.id,username, COUNT(users.id) As total_likes_by_user
FROM users
JOIN likes ON users.id = likes.user_id
GROUP BY users.id
HAVING total_likes_by_user = (SELECT COUNT(*) FROM photos);


#ERD

## Entities
- User (userId, name, phoneNum)
- Buyer (userId)
- Seller (userId)
- Bank Card (cardNumber, userId, bank, expiryDate)
- Credit Card (cardNumber, organization)
- Debit Card (cardNumber)
- Store (sid, name, startTime, customerGrade, streetAddr, city, province)
- Product (pid, sid, name, brand, type, amount, price, colour, customerReview, modelNumber)
- Order Item (itemid, pid, price, creationTime)
- Order (orderNumber, creationTime, paymentStatus, totalAmount)
- Address (addrid, userid, name, city, postalCode, streetAddr, province, contactPhoneNumber)

## Relationships 
- Manage (userid, sid, SetupTime) (userid ref Seller, sid ref Store)
- Save to Shopping Cart (userid, pid, quantity, addtime) (userid ref Buyer, pid ref Product)
- Contain (orderNumber, itemid, quantity) (orderNumber ref Order, itemid ref Order Item)
- Deliver To (addrid, orderNumber, TimeDelivered) (addrid ref Address, orderNumber ref Order)
- Payment (C.cardNumber, orderNumber, payTime) (C.cardNumber ref Credit Card, orderNumber ref Order)

-- Q2: Cinema Ticket Sales Report

import System.IO

type Sale = (String, Int)

-- Sum category tickets
sumCategory :: String -> [Sale] -> Int
sumCategory _ [] = 0
sumCategory cat ((c, q):xs)
    | cat == c  = q + sumCategory cat xs
    | otherwise = sumCategory cat xs

-- Calculate total revenue
revenue :: [Sale] -> Int
revenue [] = 0
revenue ((c, q):xs) =
    let price = case c of
                    "Adult"  -> 12
                    "Child"  -> 8
                    "Senior" -> 10
                    _        -> 0
    in q*price + revenue xs

-- Parse "Category:Quantity"
parseSale :: String -> Sale
parseSale str =
    let (c:q:_) = wordsWhen (==':') str
    in (c, read q)

-- String split utility
wordsWhen :: (Char -> Bool) -> String -> [String]
wordsWhen p s = case dropWhile p s of
    "" -> []
    s' -> w : wordsWhen p s''
        where (w, s'') = break p s'

main :: IO ()
main = do
    putStrLn "Enter sales records (Category:Quantity). Blank line to stop:"
    input <- getContents
    let userData = filter (/= "") (lines input)
    let sales = if null userData
                   then [("Adult",3),("Child",5),("Senior",2),("Adult",2)]
                   else map parseSale userData

    let adultTotal  = sumCategory "Adult" sales
    let childTotal  = sumCategory "Child" sales
    let seniorTotal = sumCategory "Senior" sales
    let totalRev    = revenue sales

    putStrLn "---- Cinema Sales Report ----"
    putStrLn $ "Adults: " ++ show adultTotal
    putStrLn $ "Children: " ++ show childTotal
    putStrLn $ "Seniors: " ++ show seniorTotal
    putStrLn $ "Total Revenue: $" ++ show totalRev

{- 
Sample Input:
Adult:3
Child:5
Senior:2
Adult:2

(Or run without typing → uses same hardcoded data)
-}

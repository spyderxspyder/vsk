-- Q1: Hospital Patient Records Analysis

import System.IO

type Patient = (String, Int, Int)

-- Count patients by reason codes recursively
countReasons :: [Patient] -> (Int, Int, Int)
countReasons [] = (0,0,0)
countReasons ((_, _, reason):xs) =
    let (c1, c2, c3) = countReasons xs
    in case reason of
        1 -> (c1+1, c2, c3)
        2 -> (c1, c2+1, c3)
        3 -> (c1, c2, c3+1)
        _ -> (c1, c2, c3)

-- Count adults (18+)
countAdults :: [Patient] -> Int
countAdults [] = 0
countAdults ((_, age, _):xs)
    | age >= 18 = 1 + countAdults xs
    | otherwise = countAdults xs

-- Parse input line "Name:Age:Reason"
parsePatient :: String -> Patient
parsePatient str =
    let (n:a:r:_) = wordsWhen (==':') str
    in (n, read a, read r)

-- Split string utility
wordsWhen :: (Char -> Bool) -> String -> [String]
wordsWhen p s = case dropWhile p s of
    "" -> []
    s' -> w : wordsWhen p s''
        where (w, s'') = break p s'

main :: IO ()
main = do
    putStrLn "Enter patient records (Name:Age:Reason). Blank line to stop:"
    input <- getContents
    let userData = filter (/= "") (lines input)
    let patients = if null userData
                      then [("Alice",25,1),("Bob",17,2),("Charlie",40,3),("Diana",30,1)]
                      else map parsePatient userData

    let (c1, c2, c3) = countReasons patients
    let adults = countAdults patients

    putStrLn $ "General Checkup: " ++ show c1
    putStrLn $ "Emergency: " ++ show c2
    putStrLn $ "Surgery: " ++ show c3
    putStrLn $ "Total Adults: " ++ show adults


{- 
Sample Input (if typing):
Alice:25:1
Bob:17:2
Charlie:40:3
Diana:30:1

(Or run without typing → uses same hardcoded data)
-}

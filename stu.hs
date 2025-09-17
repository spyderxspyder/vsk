-- Q3: Student Academic Performance Report

import System.IO

type Student = (String, Int)

-- Classify student with guards
classify :: Student -> (String, Int, String)
classify (name, mark)
    | mark < 40  = (name, mark, "Fail")
    | mark < 60  = (name, mark, "Pass")
    | mark < 80  = (name, mark, "Merit")
    | otherwise  = (name, mark, "Distinction")

-- Recursively classify all
classifyAll :: [Student] -> [(String, Int, String)]
classifyAll [] = []
classifyAll (s:xs) = classify s : classifyAll xs

-- Count passes (mark >= 40)
countPasses :: [Student] -> Int
countPasses [] = 0
countPasses ((_, mark):xs)
    | mark >= 40 = 1 + countPasses xs
    | otherwise  = countPasses xs

-- Parse "Name:Mark"
parseStudent :: String -> Student
parseStudent str =
    let (n:m:_) = wordsWhen (==':') str
    in (n, read m)

-- String split utility
wordsWhen :: (Char -> Bool) -> String -> [String]
wordsWhen p s = case dropWhile p s of
    "" -> []
    s' -> w : wordsWhen p s''
        where (w, s'') = break p s'

main :: IO ()
main = do
    putStrLn "Enter student records (Name:Mark). Blank line to stop:"
    input <- getContents
    let userData = filter (/= "") (lines input)
    let students = if null userData
                      then [("Alice",85),("Bob",35),("Charlie",60),("Diana",75)]
                      else map parseStudent userData

    let results = classifyAll students
    let passCount = countPasses students

    putStrLn "---- Student Report ----"
    mapM_ print results
    putStrLn $ "Total Passed: " ++ show passCount

{- 
Sample Input:
Alice:85
Bob:35
Charlie:60
Diana:75

(Or run without typing → uses same hardcoded data)
-}

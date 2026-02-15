# Test 1: Identify Tax Deductible Donations
# Expected: Should identify UNICEF as a deductible charity donation.
$body1 = @{
    msg = "Do I have any tax deductible donations?"
    transactions = @(
        @{
            id = "1"
            amount = 100
            currency = "USD"
            description = "UNICEF"
            date = "2024-01-01"
            category = "Charity"
        },
        @{
            id = "2"
            amount = 50
            currency = "USD"
            description = "Local Food Bank"
            date = "2024-01-15"
            category = "Charity"
        },
        @{
            id = "3"
            amount = 1200
            currency = "USD"
            description = "Rent"
            date = "2024-01-01"
            category = "Housing"
        }
    )
} | ConvertTo-Json -Depth 3

Write-Host "`n=== TEST 1: Tax Deductions ===" -ForegroundColor Cyan
$response1 = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -Headers @{ "X-Analyst-Type" = "tax" } `
  -ContentType "application/json" `
  -Body $body1

# Print the full response text properly
$response1 | Select-Object -ExpandProperty answer


# Test 2: High Spending Alert (Budget Coach)
# Expected: Should warn about the high dining expense.
$body2 = @{
    msg = "Am I spending too much on food?"
    transactions = @(
        @{
            id = "4"
            amount = 250
            currency = "USD"
            description = "Fancy Steakhouse"
            date = "2024-02-14"
            category = "Dining"
        },
        @{
            id = "5"
            amount = 15
            currency = "USD"
            description = "McDonalds"
            date = "2024-02-10"
            category = "Dining"
        }
    )
} | ConvertTo-Json -Depth 3

Write-Host "`n=== TEST 2: Budget Coach Check ===" -ForegroundColor Cyan
$response2 = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -Headers @{ "X-Analyst-Type" = "coach" } `
  -ContentType "application/json" `
  -Body $body2

$response2 | Select-Object -ExpandProperty answer


# Test 3: Investment Advice (Advisor)
# Expected: Should give general advice based on the savings deposit.
$body3 = @{
    msg = "I have some extra cash, what should I do with it?"
    transactions = @(
        @{
            id = "6"
            amount = 5000
            currency = "USD"
            description = "Bonus Check"
            date = "2024-02-01"
            category = "Income"
        }
    )
} | ConvertTo-Json -Depth 3

Write-Host "`n=== TEST 3: Investment Advice ===" -ForegroundColor Cyan
$response3 = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -Headers @{ "X-Analyst-Type" = "advisor" } `
  -ContentType "application/json" `
  -Body $body3

$response3 | Select-Object -ExpandProperty answer

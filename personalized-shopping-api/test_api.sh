#!/bin/bash

echo "=== Testing Personalized Shopping API ==="
echo ""

API_URL="http://localhost:8000"

# Test 1: Health Check
echo "1. Health Check"
echo "GET $API_URL/health"
curl -s $API_URL/health | jq '.'
echo ""
echo ""

# Test 2: Get Customer Profile
echo "2. Get Customer Profile (Kenneth Martinez - ID: 887)"
echo "GET $API_URL/api/v1/customers/887/profile"
curl -s $API_URL/api/v1/customers/887/profile | jq '{customer_name: .customer_name, total_purchases: .total_purchases, avg_purchase_price: .avg_purchase_price, favorite_categories: .favorite_categories, price_segment: .price_segment}'
echo ""
echo ""

# Test 3: Find Similar Customers
echo "3. Find Similar Customers"
echo "GET $API_URL/api/v1/customers/887/similar?top_k=5"
curl -s "$API_URL/api/v1/customers/887/similar?top_k=5" | jq '[.similar_customers[] | {customer_name: .customer_name, similarity_score: .similarity_score, common_categories: .common_categories}]'
echo ""
echo ""

# Test 4: Get Product Reviews
echo "4. Get Product Reviews (Laptop - ID: 291)"
echo "GET $API_URL/api/v1/products/291/reviews"
curl -s $API_URL/api/v1/products/291/reviews | jq '{product_name: .product_name, review_count: .review_count, avg_rating: .avg_rating, avg_sentiment: .avg_sentiment, reviews: [.reviews[0:2][] | {customer_name: .customer_name, rating: .rating, sentiment_score: .sentiment_score}]}'
echo ""
echo ""

# Test 5: Get Personalized Recommendations
echo "5. Get Personalized Recommendations for Kenneth Martinez"
echo "POST $API_URL/api/v1/recommendations/personalized"
curl -s -X POST $API_URL/api/v1/recommendations/personalized \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What else would Kenneth Martinez like based on his purchase history?",
    "customer_name": "Kenneth Martinez",
    "top_n": 5,
    "include_reasoning": true
  }' | jq '{
    customer: {
      name: .customer_profile.customer_name,
      total_purchases: .customer_profile.total_purchases,
      avg_price: .customer_profile.avg_purchase_price,
      favorite_categories: .customer_profile.favorite_categories,
      price_segment: .customer_profile.price_segment
    },
    recommendations: [.recommendations[] | {
      product_name: .product_name,
      category: .product_category,
      price: .avg_price,
      score: .recommendation_score,
      reason: .reason,
      similar_customer_count: .similar_customer_count,
      avg_sentiment: .avg_sentiment,
      source: .source
    }],
    reasoning: .reasoning,
    confidence_score: .confidence_score,
    processing_time_ms: .processing_time_ms,
    similar_customers_analyzed: .similar_customers_analyzed,
    products_considered: .products_considered,
    agent_execution_order: .agent_execution_order
  }'

echo ""
echo ""
echo "=== All Tests Complete ==="

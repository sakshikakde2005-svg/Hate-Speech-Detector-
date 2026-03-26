from transformers import pipeline

classifier = pipeline("text-classification")

comment = input("Enter a comment: ")

result = classifier(comment)

print("Prediction:", result)
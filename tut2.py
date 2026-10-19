 # Load dataset
breast_cancer = load_breast_cancer()

# Create feature DataFrame
X = pd.DataFrame(
    breast_cancer.data,
    columns=breast_cancer.feature_names
)

# Create categorical target
y = pd.Categorical.from_codes(
    breast_cancer.target,
    breast_cancer.target_names
)

# Convert target labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Dataset shape:", X.shape)
print("Target classes:", encoder.classes_)

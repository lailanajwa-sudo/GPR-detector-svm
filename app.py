# Update your SVM training line to this:
# 1=Cavity, 2=Brick, 3=Metal
# We give Brick a weight of 5.0 so the AI stops ignoring it.
from sklearn.svm import SVC

model = SVC(
    kernel='rbf', 
    C=100.0, 
    probability=True, 
    class_weight={1: 1.0, 2: 5.0, 3: 1.0} 
)
model.fit(X_scaled, y)
# Save and download these new .pkl files

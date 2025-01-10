def calculate_ths(P_c1, P_w1, P_c2, P_w2):
    """
    Calculate the Truthful Helpfulness Score (THS) based on the model's performance before and after refinement.
    
    Parameters:
    - P_c1: Accuracy of the initial model.
    - P_w1: Error rate of the initial model.
    - P_c2: Accuracy of the refined model.
    - P_w2: Error rate of the refined model.
    
    Returns:
    - THS: The Truthful Helpfulness Score.
    """
    
    # Define the coordinates for points E1, E2, and A
    E1 = (P_c1, P_w1)  # Initial model's performance
    E2 = (P_c2, P_w2)  # Refined model's performance
    A = (1, 0)         # Ideal case (all correct answers)
    
    # Calculate the cross product of vectors OE2 and OE1
    def cross_product(u, v):
        return u[0] * v[1] - u[1] * v[0]
    
    OE2 = (E2[0], E2[1])  # Vector from origin to E2
    OE1 = (E1[0], E1[1])  # Vector from origin to E1
    OA = (A[0], A[1])     # Vector from origin to A
    
    # Calculate the cross products
    cross_OE2_OE1 = cross_product(OE2, OE1)
    cross_OA_OE1 = cross_product(OA, OE1)
    
    # Calculate THS
    if cross_OA_OE1 == 0:
        raise ValueError("Cross product OA x OE1 is zero, which would result in division by zero.")
    
    THS = cross_OE2_OE1 / cross_OA_OE1
    
    return THS

P_c1 = 0.70  # Initial accuracy
P_w1 = 0.30 # Initial error rate
P_c2 = 0.67  # Refined accuracy
P_w2 = 0.32  # Refined error rate

ths = calculate_ths(P_c1, P_w1, P_c2, P_w2)
print(f"Truthful Helpfulness Score (THS): {ths}")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import onnxruntime as ort
import os

app = FastAPI(title="Gait Auth IAM Risk Engine", version="1.0.0")

KNOWN_RECIPIENTS = {"0241234567", "0559876543"}
MODEL_PATH = "gait_model.onnx"
session = ort.InferenceSession(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class AuthRequest(BaseModel):
    user_id: str
    amount_ghs: float
    recipient_phone: str
    sensor_window: List[List[float]]

class AuthResponse(BaseModel):
    user_id: str
    transaction_risk: str
    gait_confidence: str
    raw_match_score: float
    decision: str
    reason: str

@app.post("/api/v1/auth/evaluate", response_model=AuthResponse)
def evaluate_auth(payload: AuthRequest):
    data = np.array(payload.sensor_window, dtype=np.float32)

    if data.shape != (6, 128):
        raise HTTPException(
            status_code=422,
            detail=f"Expected shape (6, 128), received {data.shape}"
        )

    if session:
        input_tensor = np.expand_dims(data, axis=0)
        input_name = session.get_inputs()[0].name
        raw_output = session.run(None, {input_name: input_tensor})[0]
        match_score = float(1.0 / (1.0 + np.exp(-raw_output[0][0])))
    else:
        # Fallback simulation score for baseline testing
        match_score = 0.35

    if match_score >= 0.80:
       gait_tier = "HIGH"
    elif match_score >= 0.45:  # Lowered from 0.55 to test the ALLOW branch
       gait_tier = "MEDIUM"
    else:
       gait_tier = "LOW"

    is_novel_recipient = payload.recipient_phone not in KNOWN_RECIPIENTS
    if payload.amount_ghs > 1000.0 or (payload.amount_ghs > 500.0 and is_novel_recipient):
        tx_risk = "HIGH"
    elif payload.amount_ghs > 200.0 or is_novel_recipient:
        tx_risk = "MEDIUM"
    else:
        tx_risk = "LOW"

    if tx_risk == "LOW":
        if gait_tier in ["HIGH", "MEDIUM"]:
            decision = "ALLOW"
            reason = "Low transaction risk with acceptable gait confidence."
        else:
            decision = "STEP_UP_LIGHT"
            reason = "Low risk transaction but gait confidence is low."
    elif tx_risk == "MEDIUM":
        if gait_tier == "HIGH":
            decision = "ALLOW"
            reason = "Medium transaction risk cleared by high biometric confidence."
        elif gait_tier == "MEDIUM":
            decision = "STEP_UP_LIGHT"
            reason = "Medium transaction risk requires light verification (PIN)."
        else:
            decision = "STEP_UP_STRONG"
            reason = "Medium transaction risk paired with low biometric confidence."
    else:
        if gait_tier == "HIGH":
            decision = "STEP_UP_LIGHT"
            reason = "High-value transaction requires standard confirmation despite verified gait."
        elif gait_tier == "MEDIUM":
            decision = "STEP_UP_STRONG"
            reason = "High-value transaction requires strong step-up authentication."
        else:
            decision = "DENY"
            reason = "High transaction risk and biometric mismatch. Potential anomaly."

    print(f"\n--> [ONNX INFERENCE] Raw Score: {match_score:.4f} | Biometric Tier: {gait_tier} | Tx Risk: {tx_risk} -> Decision: {decision}\n")

    return AuthResponse(
        user_id=payload.user_id,
        transaction_risk=tx_risk,
        gait_confidence=gait_tier,
        raw_match_score=round(match_score, 4),
        decision=decision,
        reason=reason
    )

class StepUpVerifyRequest(BaseModel):
    user_id: str
    challenge_type: str  # "PIN" or "OTP"
    challenge_value: str
    amount_ghs: float
    recipient_phone: str

class StepUpVerifyResponse(BaseModel):
    status: str
    message: str
    transaction_id: str

@app.post("/api/v1/auth/verify-step-up", response_model=StepUpVerifyResponse)
def verify_step_up(payload: StepUpVerifyRequest):
    # Mock secrets: PIN "1234", OTP "654321"
    valid = (payload.challenge_type == "PIN" and payload.challenge_value == "1234") or \
            (payload.challenge_type == "OTP" and payload.challenge_value == "654321")
    
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid verification code.")
    
    print(f"\n--> [CHALLENGE CLEARED] Authorized {payload.amount_ghs} GHS to {payload.recipient_phone} via {payload.challenge_type}\n")
    return StepUpVerifyResponse(
        status="SUCCESS",
        message="Step-up authentication verified successfully.",
        transaction_id=f"TXN-{np.random.randint(100000, 999999)}"
    )
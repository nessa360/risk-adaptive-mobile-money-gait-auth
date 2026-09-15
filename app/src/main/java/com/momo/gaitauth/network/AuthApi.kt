package com.momo.gaitauth.network

import com.google.gson.annotations.SerializedName
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

data class TransactionPayload(
    @SerializedName("user_id") val userId: String,
    @SerializedName("amount_ghs") val amountGhs: Double,
    @SerializedName("recipient_phone") val recipientPhone: String,
    @SerializedName("sensor_window") val sensorWindow: List<List<Float>>
)

enum class AuthDecision {
    ALLOW,
    STEP_UP_LIGHT,
    STEP_UP_STRONG,
    DENY
}

data class DecisionResponse(
    @SerializedName("user_id") val userId: String,
    @SerializedName("transaction_risk") val transactionRisk: String,
    @SerializedName("gait_confidence") val gaitConfidence: String,
    @SerializedName("raw_match_score") val rawMatchScore: Float,
    @SerializedName("decision") val decision: AuthDecision,
    @SerializedName("reason") val reason: String
)

data class StepUpVerifyRequest(
    @SerializedName("user_id") val userId: String,
    @SerializedName("challenge_type") val challengeType: String, // "PIN" or "OTP"
    @SerializedName("challenge_value") val challengeValue: String,
    @SerializedName("amount_ghs") val amountGhs: Double,
    @SerializedName("recipient_phone") val recipientPhone: String
)

data class StepUpVerifyResponse(
    @SerializedName("status") val status: String,
    @SerializedName("message") val message: String,
    @SerializedName("transaction_id") val transactionId: String
)

interface GaitAuthApiService {
    @POST("/api/v1/auth/evaluate")
    suspend fun evaluateTransaction(
        @Body payload: TransactionPayload
    ): Response<DecisionResponse>

    @POST("/api/v1/auth/verify-step-up")
    suspend fun verifyStepUp(
        @Body payload: StepUpVerifyRequest
    ): Response<StepUpVerifyResponse>
}
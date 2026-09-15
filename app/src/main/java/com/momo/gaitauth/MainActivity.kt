package com.momo.gaitauth

import android.content.Intent
import android.os.Bundle
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.momo.gaitauth.network.ApiClient
import com.momo.gaitauth.network.AuthDecision
import com.momo.gaitauth.network.StepUpVerifyRequest
import com.momo.gaitauth.network.TransactionPayload
import com.momo.gaitauth.sensor.GaitSensorService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private var pendingAmount: Double = 0.0
    private var pendingRecipient: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 1. Start continuous sensing foreground service immediately
        val serviceIntent = Intent(this, GaitSensorService::class.java)
        startService(serviceIntent)

        val btnSend = findViewById<Button>(R.id.btnSendMoney)
        val etAmount = findViewById<EditText>(R.id.etAmount)
        val etRecipient = findViewById<EditText>(R.id.etRecipient)

        btnSend.setOnClickListener {
            val amount = etAmount.text.toString().toDoubleOrNull() ?: 0.0
            val recipient = etRecipient.text.toString().trim()

            if (recipient.isNotEmpty() && amount > 0.0) {
                pendingAmount = amount
                pendingRecipient = recipient
                initiateTransaction(amount, recipient)
            } else {
                Toast.makeText(this, "Enter valid recipient and amount", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun initiateTransaction(amount: Double, recipient: String) {
        // Guard check: ensure enough walk cycles have been recorded
        if (!GaitSensorService.gaitBuffer.isFull()) {
            showStepUpDialog("Collecting behavioural baseline. Please authenticate with PIN.", isStrong = false)
            return
        }

        // Convert (6 x 128) array to List<List<Float>> for JSON transmission
        val rawArray = GaitSensorService.gaitBuffer.toModelInputArray()
        val bufferList: List<List<Float>> = rawArray.map { it.toList() }

        val payload = TransactionPayload(
            userId = "user_gh_01",
            amountGhs = amount,
            recipientPhone = recipient,
            sensorWindow = bufferList
        )

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val response = ApiClient.service.evaluateTransaction(payload)
                withContext(Dispatchers.Main) {
                    if (response.isSuccessful && response.body() != null) {
                        handleDecision(response.body()!!.decision, response.body()!!.reason)
                    } else {
                        Toast.makeText(this@MainActivity, "Server rejected payload", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Network error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun handleDecision(decision: AuthDecision, reason: String) {
        when (decision) {
            AuthDecision.ALLOW -> {
                showSuccessDialog("Transaction Approved! Continuous gait verified. ($reason)")
            }
            AuthDecision.STEP_UP_LIGHT -> {
                showStepUpDialog("Step-Up (Light): $reason", isStrong = false)
            }
            AuthDecision.STEP_UP_STRONG -> {
                showStepUpDialog("Step-Up (Strong): $reason. OTP sent to registered SIM.", isStrong = true)
            }
            AuthDecision.DENY -> {
                showDenyDialog("Transaction Denied: $reason")
            }
        }
    }

    private fun showStepUpDialog(msg: String, isStrong: Boolean) {
        val input = EditText(this).apply {
            hint = if (isStrong) "Enter 6-digit OTP" else "Enter 4-digit PIN"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
        }

        val dialog = AlertDialog.Builder(this)
            .setTitle(if (isStrong) "Security Challenge (High Risk)" else "Verify Identity")
            .setMessage(msg)
            .setView(input)
            .setPositiveButton("Verify & Authorise", null)
            .setNegativeButton("Cancel", null)
            .create()

        dialog.show()

        // Override positive button onClick to prevent automatic dismiss on invalid input
        dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
            val enteredValue = input.text.toString().trim()
            val expectedLength = if (isStrong) 6 else 4
            val challengeType = if (isStrong) "OTP" else "PIN"

            if (enteredValue.length != expectedLength) {
                input.error = "Requires $expectedLength digits"
                return@setOnClickListener
            }

            val request = StepUpVerifyRequest(
                userId = "user_gh_01",
                challengeType = challengeType,
                challengeValue = enteredValue,
                amountGhs = pendingAmount,
                recipientPhone = pendingRecipient
            )

            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val response = ApiClient.service.verifyStepUp(request)
                    withContext(Dispatchers.Main) {
                        if (response.isSuccessful && response.body() != null) {
                            val res = response.body()!!
                            dialog.dismiss()
                            showSuccessDialog("Payment Sent!\nTxn Ref: ${res.transactionId}\n${res.message}")
                        } else {
                            input.error = "Incorrect $challengeType. Try again."
                        }
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Network error: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    private fun showSuccessDialog(msg: String) {
        AlertDialog.Builder(this)
            .setTitle("Success")
            .setMessage(msg)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showDenyDialog(msg: String) {
        AlertDialog.Builder(this)
            .setTitle("Transaction Blocked")
            .setMessage(msg)
            .setIcon(android.R.drawable.ic_dialog_alert)
            .setPositiveButton("Dismiss", null)
            .show()
    }
}
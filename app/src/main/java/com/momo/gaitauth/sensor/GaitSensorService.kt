package com.momo.gaitauth.sensor

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

class GaitSensorService : Service(), SensorEventListener {

    companion object {
        const val CHANNEL_ID = "GaitAuthChannel"
        const val NOTIFICATION_ID = 1001
        val gaitBuffer = GaitDataBuffer(capacity = 128)
    }

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null

    private var lastAx = 0f; private var lastAy = 0f; private var lastAz = 0f
    private var lastGx = 0f; private var lastGy = 0f; private var lastGz = 0f

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        createNotificationChannel()
        val notification = buildNotification()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        registerSensors()
    }

    private fun registerSensors() {
        val samplingPeriodUs = 20_000 // 50 Hz
        accelSensor?.let {
            sensorManager.registerListener(this, it, samplingPeriodUs)
        }
        gyroSensor?.let {
            sensorManager.registerListener(this, it, samplingPeriodUs)
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        val timestamp = event.timestamp

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAx = event.values[0]
                lastAy = event.values[1]
                lastAz = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0]
                lastGy = event.values[1]
                lastGz = event.values[2]

                gaitBuffer.add(
                    IMUSample(
                        timestampNs = timestamp,
                        ax = lastAx, ay = lastAy, az = lastAz,
                        gx = lastGx, gy = lastGy, gz = lastGz
                    )
                )
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onDestroy() {
        sensorManager.unregisterListener(this)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun buildNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Continuous Security Active")
            .setContentText("Monitoring behavioural gait patterns...")
            .setSmallIcon(android.R.drawable.ic_lock_idle_lock)
            .setOngoing(true)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Gait Security Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }
}
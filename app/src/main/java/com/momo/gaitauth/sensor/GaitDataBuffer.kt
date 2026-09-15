package com.momo.gaitauth.sensor

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

data class IMUSample(
    val timestampNs: Long,
    val ax: Float, val ay: Float, val az: Float,
    val gx: Float, val gy: Float, val gz: Float
)

class GaitDataBuffer(val capacity: Int = 128) {
    private val buffer = ArrayList<IMUSample>(capacity)
    private val lock = ReentrantLock()

    fun add(sample: IMUSample) = lock.withLock {
        if (buffer.size >= capacity) {
            buffer.removeAt(0)
        }
        buffer.add(sample)
    }

    fun isFull(): Boolean = lock.withLock { buffer.size == capacity }

    fun getSnapshot(): List<IMUSample> = lock.withLock {
        ArrayList(buffer)
    }

    fun toModelInputArray(): Array<FloatArray> = lock.withLock {
        val channels = Array(6) { FloatArray(capacity) }
        buffer.forEachIndexed { i, sample ->
            channels[0][i] = sample.ax
            channels[1][i] = sample.ay
            channels[2][i] = sample.az
            channels[3][i] = sample.gx
            channels[4][i] = sample.gy
            channels[5][i] = sample.gz
        }
        channels
    }
}
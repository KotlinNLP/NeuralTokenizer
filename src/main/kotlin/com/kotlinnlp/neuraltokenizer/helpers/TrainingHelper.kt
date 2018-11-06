/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.helpers

import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.utils.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.Shuffler
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * A helper for the training of a [NeuralTokenizer].
 *
 * @property tokenizer the [NeuralTokenizer] to train
 * @property optimizer the [NeuralTokenizerOptimizer] of the [tokenizer]
 */
class TrainingHelper(
  val tokenizer: NeuralTokenizer,
  val optimizer: NeuralTokenizerOptimizer = NeuralTokenizerOptimizer(tokenizer.model)
) {

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * The [ValidationHelper] used to validate each epoch when a not null validation dataset is passed.
   */
  private val validationHelper = ValidationHelper(this.tokenizer.model)

  /**
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = 0.0

  /**
   * The gold classification of the current segment.
   */
  private lateinit var segmentGoldClassification: List<Int>

  /**
   * Train the [tokenizer] using the chars classifications of the [trainingSet] as reference of correct predictions.
   *
   * @param trainingSet the [Dataset] to train the [tokenizer]
   * @param batchSize the size of the training batches (default 1)
   * @param epochs number of epochs
   * @param validationSet the [Dataset] used to validate each epoch (default null)
   * @param shuffler the [Shuffler] to shuffle the training sentences before each epoch (default null)
   * @param modelFilename the name of the file in which to save the best trained model
   */
  fun train(trainingSet: Dataset,
            batchSize: Int = 1,
            epochs: Int = 3,
            validationSet: Dataset? = null,
            shuffler: Shuffler? = null,
            modelFilename: String? = null) {

    println("-- START TRAINING ON %d SENTENCES".format(trainingSet.size))

    this.resetValidationStats()

    this.initEmbeddings(trainingSet)

    (0 until epochs).forEach { i ->

      println("\nEpoch ${i + 1} of $epochs")

      this.startTiming()

      val mergedSentences = mergeDataset(
        dataset = if (shuffler != null) shuffleDataset(dataset = trainingSet, shuffler = shuffler) else trainingSet)

      this.trainEpoch(text = mergedSentences.first, goldClassifications = mergedSentences.second, batchSize = batchSize)

      println("Elapsed time: %s".format(this.formatElapsedTime()))

      if (validationSet != null) {
        this.validateAndSaveModel(validationSet = validationSet, modelFilename = modelFilename)
      }
    }
  }

  /**
   * Initialize the Embeddings of the [tokenizer] model, associating them to the chars contained in the given
   * [trainingSet].
   *
   * @param trainingSet the dataset to train the [tokenizer]
   */
  private fun initEmbeddings(trainingSet: Dataset) {

    trainingSet.forEach { (sentence, _) ->
      sentence.forEach { char ->
        if (char !in this.tokenizer.model.embeddings) {
          this.tokenizer.model.embeddings.set(key = char)
        }
      }
    }
  }

  /**
   * Train the [tokenizer] on one epoch, using the [goldClassifications] as reference of correct predictions.
   *
   * @param text the text to tokenize
   * @param goldClassifications an array containing the correct classification of each character
   * @param batchSize the size of the training batches
   */
  private fun trainEpoch(text: String, goldClassifications: CharsClassification, batchSize: Int) {

    require(text.length == goldClassifications.size)

    var examplesCount = 0

    this.optimizer.newEpoch()

    this.forEachSegment(text = text, goldClassifications = goldClassifications) { (startIndex, endIndex) ->

      examplesCount++

      this.learnFromExample(text = text, start = startIndex, length = endIndex - startIndex)

      if (examplesCount % batchSize == 0) {
        this.endOfBatch()
      }
    }

    if (examplesCount % batchSize > 0) { // last batch with remaining examples
      this.endOfBatch()
    }
  }

  /**
   * Iterate over the segments of training extracted shifting the text using the [goldClassifications].
   * Before returning the segment, it is classified from the tokenizer and the [segmentGoldClassification] is set.
   *
   * @param text the text to tokenize
   * @param goldClassifications the gold classification of each character of the [text]
   * @param callback a callback called for each segment (it takes a pair of <start(inclusive), end(exclusive)> indices
   *                 as argument)
   */
  private fun forEachSegment(text: String,
                              goldClassifications: CharsClassification,
                              callback: (Pair<Int, Int>) -> Unit) {

    val progress = ProgressIndicatorBar(total = text.length)
    var startIndex = 0

    while (startIndex < text.length) {

      val endIndex: Int = minOf(startIndex + this@TrainingHelper.tokenizer.model.maxSegmentSize, text.length)

      this@TrainingHelper.segmentGoldClassification = goldClassifications.subList(startIndex, endIndex)

      callback(Pair(startIndex, endIndex))

      val shiftCharIndex = this@TrainingHelper.getShiftCharIndex()

      progress.tick(amount = shiftCharIndex + 1)

      startIndex += shiftCharIndex + 1
    }
  }

  /**
   * @return the index of the last char to remove from the segment shifting it to left
   */
  private fun getShiftCharIndex(): Int {

    val segmentSize : Int = this.segmentGoldClassification.size
    var lastSentenceBoundary: Int = segmentSize - 1

    while (lastSentenceBoundary >= 0 &&
      this.segmentGoldClassification[lastSentenceBoundary] != 1) lastSentenceBoundary--

    return if (lastSentenceBoundary >= 0)
      lastSentenceBoundary
    else
      this.getMiddleTokenBoundary().let { if (it >= 0) it else segmentSize / 2 }
  }

  /**
   * Return the index of the boundary of the token that crosses the middle of the gold classification of the current
   * segment.
   * Search in the second half of the segment first, starting from the middle. If any boundary is found the search
   * continues in the first half of the segment, starting from the middle again.
   *
   * @return the index of the boundary of the middle token if any, -1 otherwise
   */
  private fun getMiddleTokenBoundary(): Int {

    val segmentSize : Int = this.segmentGoldClassification.size
    val halfSegmentSize: Int = segmentSize / 2
    var middleTokenBoundary: Int = halfSegmentSize

    // search in the second half
    while (middleTokenBoundary < segmentSize &&
      this.segmentGoldClassification[middleTokenBoundary] != 0) middleTokenBoundary++

    if (middleTokenBoundary >= segmentSize) {

      middleTokenBoundary = halfSegmentSize

      // search in the first half
      while (middleTokenBoundary > 0 &&
        this.segmentGoldClassification[middleTokenBoundary] != 0) middleTokenBoundary--
    }

    return if (middleTokenBoundary in 0 until segmentSize) middleTokenBoundary else -1
  }

  /**
   * Learn from the given segment, comparing its gold classification with the one of the [tokenizer] and accumulate
   * the propagated errors.
   *
   * @param text the whole text to tokenize
   * @param start the start index of the segment
   * @param length the length of the segment
   */
  private fun learnFromExample(text: String, start: Int, length: Int) {

    this.optimizer.newExample()

    this.backward(segmentClassification = this.tokenizer.classifyChars(text = text, start = start, length = length))

    this.accumulateErrors(segment = text.subSequence(start, start + length))
  }

  /**
   * Accumulate the parameters errors into the optimizer.
   *
   * @param segment the segment used for the last backward
   */
  private fun accumulateErrors(segment: CharSequence) {

    this.optimizer.charsEncoderOptimizer.accumulate(this.tokenizer.charsEncoder.getParamsErrors(copy = false))
    this.optimizer.boundariesClassifierOptimizer.accumulate(this.tokenizer.boundariesClassifier.getParamsErrors(copy = false))

    this.tokenizer.charsEncoder.getInputErrors(copy = false).forEachIndexed { i, errors ->
      this.optimizer.embeddingsOptimizer.accumulate(
        embeddingKey = segment[i],
        errors = errors.getRange(0, errors.length - this.tokenizer.model.addingFeaturesSize)
      )
    }
  }

  /**
   * Catch the event that corresponds to the end of a training batch.
   */
  private fun endOfBatch() {
    this.optimizer.newBatch()
    this.optimizer.update()
  }

  /**
   * Backward of the sub-networks of the [tokenizer] starting from the errors generated comparing the current
   * [segmentClassification] with the [segmentGoldClassification].
   *
   * @param segmentClassification the classification of the current segment
   */
  private fun backward(segmentClassification: List<DenseNDArray>) {

    this.backwardBoundariesClassifier(
      segmentClassification = segmentClassification,
      goldSegmentClassification = this.segmentGoldClassification)

    this.tokenizer.charsEncoder.backward(this.tokenizer.boundariesClassifier.getInputErrors(copy = false))
  }

  /**
   * Backward of the boundaries classifier of the [tokenizer] starting from the errors generated comparing the current
   * [segmentClassification] with the [segmentGoldClassification].
   *
   * @param segmentClassification the classification of the current segment
   */
  private fun backwardBoundariesClassifier(segmentClassification: List<DenseNDArray>,
                                           goldSegmentClassification: List<Int>) {

    this.tokenizer.boundariesClassifier.backward(
      List(
        size = goldSegmentClassification.size,
        init = { i ->
          val charClassification: DenseNDArray = segmentClassification[i]
          val goldClass: Int = goldSegmentClassification[i]

          charClassification[goldClass] = charClassification[goldClass] - 1

          charClassification
        }
      )
    )
  }

  /**
   * Validate the [tokenizer] on the [validationSet] and save its best model to [modelFilename].
   *
   * @param validationSet the validation dataset to validate the [tokenizer]
   * @param modelFilename the name of the file in which to save the best model of the [tokenizer] (default = null)
   */
  private fun validateAndSaveModel(validationSet: Dataset, modelFilename: String?) {

    val accuracy = this.validateEpoch(validationSet)

    if (modelFilename != null && accuracy > this.bestAccuracy) {

      this.bestAccuracy = accuracy

      this.tokenizer.model.dump(FileOutputStream(File(modelFilename)))

      println("NEW BEST ACCURACY! Model saved to \"$modelFilename\"")
    }
  }

  /**
   * Validate the [tokenizer] after trained it on an epoch.
   *
   * @param validationSet the validation dataset to validate the [tokenizer]
   *
   * @return the current accuracy of the [tokenizer]
   */
  private fun validateEpoch(validationSet: Dataset): Double {

    println("Epoch validation on %d sentences".format(validationSet.size))

    val stats: ValidationHelper.EvaluationStats = this.validationHelper.validate(validationSet)

    println("Tokens accuracy     ->   Precision: %.2f%%  |  Recall: %.2f%%  |  F1 Score: %.2f%%"
      .format(100.0 * stats.tokens.precision, 100.0 * stats.tokens.recall, 100.0 * stats.tokens.f1Score))

    println("Sentences accuracy  ->   Precision: %.2f%%  |  Recall: %.2f%%  |  F1 Score: %.2f%%"
      .format(100.0 * stats.sentences.precision, 100.0 * stats.sentences.recall, 100.0 * stats.sentences.f1Score))

    return this.getAccuracy(stats)
  }

  /**
   * Reset the stats saved during the previous validations.
   */
  private fun resetValidationStats() {

    this.bestAccuracy = 0.0
  }

  /**
   * Calculate the accuracy of the model, giving an higher weight to the sentences metric.
   *
   * @param stats the validation statistics given by the [ValidationHelper]
   *
   * @return the accuracy of the [tokenizer]
   */
  private fun getAccuracy(stats: ValidationHelper.EvaluationStats): Double {

    return stats.tokens.f1Score * Math.pow(stats.sentences.f1Score, 0.5)
  }

  /**
   * Start registering time.
   */
  private fun startTiming() {
    this.startTime = System.currentTimeMillis()
  }

  /**
   * @return the formatted string with elapsed time in seconds and minutes.
   */
  private fun formatElapsedTime(): String {

    val elapsedTime = System.currentTimeMillis() - this.startTime
    val elapsedSecs = elapsedTime / 1000.0

    return "%.3f s (%.1f min)".format(elapsedSecs, elapsedSecs / 60.0)
  }
}

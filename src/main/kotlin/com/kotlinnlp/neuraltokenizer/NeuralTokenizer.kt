/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

import com.kotlinnlp.neuraltokenizer.utils.AbbreviationsContainer
import com.kotlinnlp.neuraltokenizer.utils.abbreviations
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import kotlin.coroutines.experimental.buildSequence

/**
 * Neural Tokenizer.
 *
 * @property model the model for the sub-networks of this [NeuralTokenizer]
 */
class NeuralTokenizer(val model: NeuralTokenizerModel) {

  /**
   * The [BiRNNEncoder] used to encode the characters of a segment.
   */
  val charsEncoder = BiRNNEncoder<DenseNDArray>(this.model.biRNN)

  /**
   * The processor of the boundariesNetworkModel.
   */
  val boundariesClassifier = BatchFeedforwardProcessor<DenseNDArray>(this.model.boundariesNetworkModel)

  /**
   * A Boolean indicating if the language uses the "scriptio continua" style (writing without spaces).
   */
  private val useScriptioContinua: Boolean = this.model.language in setOf("zh", "ja", "th")

  /**
   * The sentences resulting from the tokenization of a text.
   */
  private var sentences = ArrayList<Sentence>()

  /**
   * The currently buffered token.
   */
  private var curTokenBuffer = StringBuffer()

  /**
   * The currently buffered sentence.
   */
  private val curSentenceBuffer = StringBuffer()

  /**
   * The list of completed tokens of the currently buffered sentence.
   */
  private var curSentenceTokens: ArrayList<Token> = arrayListOf()

  /**
   * Tokenize the text splitting it in [Sentence]s and [Token]s.
   *
   * @param text the text to tokenize
   *
   * @return the list of sentences which compose the [text], each containing the list of tokens
   */
  fun tokenize(text: String): ArrayList<Sentence> {

    this.sentences = ArrayList()

    this.loopSegments(text).forEach { (startIndex, endIndex) ->
      this.processSegment(text = text, start = startIndex, end = endIndex)
    }

    return this.sentences
  }

  /**
   * @param text the whole text to tokenize
   * @param start the start index of the focus segment
   * @param length the length of the focus segment
   *
   * @return a list with the classification array of each character
   *         (0 = token boundary follows, 1 = sequence boundary follows, 2 = no boundary follows)
   */
  fun classifyChars(text: String, start: Int, length: Int): List<DenseNDArray> =
    this.boundariesClassifier.forward(
      this.charsEncoder.encode(
        sequence = this.charsToEmbeddings(
          text = text,
          start = start,
          length = length)))

  /**
   * Loop over the segments of text.
   *
   * @param text the text to tokenize
   *
   * @return a Pair containing the start (inclusive) and end (exclusive) indices of the current segment
   */
  private fun loopSegments(text: String) = buildSequence {

    var startIndex = 0

    while (startIndex < text.length) {

      val endIndex: Int = minOf(startIndex + this@NeuralTokenizer.model.maxSegmentSize, text.length)

      yield(Pair(startIndex, endIndex))

      val lastTokenIndex: Int = this@NeuralTokenizer.getLastTokenEndIndex()

      startIndex = lastTokenIndex + this@NeuralTokenizer.curTokenBuffer.length + 1
    }
  }

  /**
   * @return the end index of the last token added to the current buffer
   */
  private fun getLastTokenEndIndex(): Int = when {
    this.curSentenceTokens.size > 0 -> this.curSentenceTokens.last().endAt // new tokens added
    this.sentences.size > 0 -> this@NeuralTokenizer.sentences.last().endAt // new sentences added
    else -> 0 // first token, no new tokens or sentences added (no boundaries found)
  }

  /**
   * Process the segment of [text] between the indices [start] and [end].
   *
   * @param text the text to tokenize
   * @param start the start index of the segment (inclusive)
   * @param end the end index of the segment (exclusive)
   */
  private fun processSegment(text: String, start: Int, end: Int) {

    val charsClassification = this.classifyChars(text = text, start = start, length = end - start)
    val prevSentencesCount: Int = this.sentences.size
    val sentencePrevTokensCount: Int = this.curSentenceTokens.size

    charsClassification.forEachIndexed { i, charClassification ->
      val textIndex: Int = start + i

      this.processChar(
        char = text[textIndex],
        nextChar = if (textIndex < text.lastIndex) text[textIndex + 1] else null,
        charIndex = textIndex,
        charClass = charClassification.argMaxIndex())
    }

    this.shiftBuffer(prevSentencesCount = prevSentencesCount, sentencePrevTokensCount = sentencePrevTokensCount)
  }

  /**
   * Shift buffers to left basing on the current prediction.
   * If new sentences are added, buffers are shifted removing all the completed sentences.
   * If only tokens are added, buffers are shifted removing the first N tokens until the one that crosses the middle of
   * the segment.
   * If neither sentences or tokens are added, buffers are shifted of an amount of chars equal to half of the max
   * segment size (defined in the model).
   *
   * @param prevSentencesCount the number of completed sentences before processing the current segment
   * @param sentencePrevTokensCount the number of completed tokens of the current sentence before processing the current
   *                                segment
   */
  private fun shiftBuffer(prevSentencesCount: Int, sentencePrevTokensCount: Int) {

    if (this.sentences.size > prevSentencesCount) {
      // New sentences added
      this.shiftBufferBySentences()

    } else {
      if (this.curSentenceTokens.isEmpty() || this.curSentenceTokens.size == sentencePrevTokensCount) {
        // No boundaries found
        this.shiftHalfBuffer()

      } else {
        // New tokens added
        this.shiftBufferByTokens(sentencePrevTokensCount = sentencePrevTokensCount)
      }
    }
  }

  /**
   * Shift buffers of an amount of chars equal to half segment.
   */
  private fun shiftHalfBuffer() {

    val halfSegmentSize: Int = this.model.maxSegmentSize / 2

    this.curSentenceBuffer.delete(this.curSentenceBuffer.length - halfSegmentSize, this.curSentenceBuffer.length)
    this.curTokenBuffer.delete(this.curTokenBuffer.length - halfSegmentSize, this.curTokenBuffer.length)
  }

  /**
   * Shift buffers of an amount equal to the first completed tokens until the one in the middle of the current segment.
   *
   * @param sentencePrevTokensCount the number of completed tokens of the current sentence before processing the current
   *                                segment
   */
  private fun shiftBufferByTokens(sentencePrevTokensCount: Int) {

    val curSegmentTokens = this.curSentenceTokens.subList(sentencePrevTokensCount, this.curSentenceTokens.size)
    val tokensIterator = curSegmentTokens.iterator()
    var tokensCharsCount = 0
    var curSegmentTokensToKeep = 0

    while (tokensIterator.hasNext() && tokensCharsCount < this.model.maxSegmentSize / 2) {
      val token: Token = tokensIterator.next()
      tokensCharsCount += token.form.length
      curSegmentTokensToKeep++
    }

    val sentencePrevLength: Int = (0 until sentencePrevTokensCount).sumBy { i -> this.curSentenceTokens[i].form.length }
    val deleteFrom: Int = sentencePrevLength + tokensCharsCount
    this.curSentenceBuffer.delete(deleteFrom, this.curSentenceBuffer.length)

    val tokensToKeep: Int = sentencePrevTokensCount + curSegmentTokensToKeep
    (tokensToKeep until this.curSentenceTokens.size).reversed().forEach { i -> this.curSentenceTokens.removeAt(i) }

    this.resetCurTokenBuffer()
  }

  /**
   * Shift buffers of an amount equal to all completed sentences (= reset buffers currently not completed).
   */
  private fun shiftBufferBySentences() {

    this.resetCurSentenceBuffer()
    this.resetCurTokenBuffer()
  }

  /**
   * Associate to each character of the sequence an embeddings vector.
   *
   * @param text the whole text to tokenize
   * @param start the start index of the focus segment
   * @param length the length of the focus segment
   *
   * @return the list of embeddings associated to the given segment (one per char)
   */
  private fun charsToEmbeddings(text: String, start: Int, length: Int) = List(
    size = length,
    init = { offset -> text.extractFeatures(start + offset) }
  )

  /**
   * @param focusIndex the index of the focus char
   *
   * @return the [DenseNDArray] of features associated to the char at [focusIndex]
   */
  private fun String.extractFeatures(focusIndex: Int): DenseNDArray {

    val char: Char = this[focusIndex]
    val embedding: DenseNDArray = char.toEmbedding()
    val features: DenseNDArray = DenseNDArrayFactory.emptyArray(
      shape = Shape(embedding.length + this@NeuralTokenizer.model.addingFeaturesSize)
    )

    // Set embedding features
    (0 until embedding.length).forEach { i -> features[i] = embedding[i] }

    val nextEndOfAbbreviation: Boolean = focusIndex < this.lastIndex && this.isEndOfAbbreviation(focusIndex + 1)

    // Set adding features (isLetter, isDigit, "end of abbreviation")
    features[features.length - 4] = if (this[focusIndex].isLetter()) 1.0 else 0.0
    features[features.length - 3] = if (this[focusIndex].isDigit()) 1.0 else 0.0
    features[features.length - 2] = if (this.isEndOfAbbreviation(focusIndex)) 1.0 else 0.0
    features[features.length - 1] = if (nextEndOfAbbreviation) 1.0 else 0.0

    return features
  }

  /**
   * Check if the char at [focusIndex] is the end of an abbreviation, looking for a match for all possible substrings
   * of this one which end with the focus char.
   *
   * @param focusIndex the index of the focus char
   *
   * @return a Boolean indicating if the char at [focusIndex] is the end of a common abbreviation
   */
  private fun String.isEndOfAbbreviation(focusIndex: Int): Boolean {

    if (this[focusIndex] == '.' && focusIndex > 0 && this@NeuralTokenizer.model.language in abbreviations) {

      val langAbbreviations: AbbreviationsContainer = abbreviations[this@NeuralTokenizer.model.language]!!

      val firstUsefulCharIndex: Int = focusIndex - minOf(focusIndex, langAbbreviations.maxLength - 1)
      var cadidateStart = focusIndex - 1 // the start index of the candidate abbreviation

      // Back to the first whitespace
      while (cadidateStart > firstUsefulCharIndex && !this[cadidateStart].isWhitespace()) { cadidateStart-- }

      if (this[cadidateStart].isWhitespace()) { // Consider only substrings delimited by a whitespace
        val candidate: String = this.substring(cadidateStart + 1..focusIndex).toLowerCase() // trim initial whitespace
        return candidate in langAbbreviations.set
      }
    }

    return false
  }

  /**
   * @return the embedding associated to this [Char]
   */
  private fun Char.toEmbedding(): DenseNDArray {
    return this@NeuralTokenizer.model.embeddings.get(key = this).array.values
  }

  /**
   * Process the [char] understanding if a token or a sentence is just ended at the given [charIndex].
   *
   * @param char the char to process
   * @param charIndex the index of the [char] within the text
   * @param charClass the predicted class of the [char]
   */
  private fun processChar(char: Char, nextChar: Char?, charIndex: Int, charClass: Int) {

    val isSpacingChar: Boolean = char.isWhitespace()

    if (isSpacingChar && this.curTokenBuffer.isNotEmpty()) { // automatically add the previously buffered token
      this.addToken(endAt = charIndex - 1, isSpace = false)
    }

    this.addToBuffers(char)

    if (nextChar == null) {
      // End of text
      this.addToken(endAt = charIndex, isSpace = isSpacingChar)
      this.addSentence(endAt = charIndex)

    } else when (charClass) {

      // token boundary follows
      0 -> if (isSpacingChar || !this.isMiddleOfWord(char, nextChar)) {
        this.addToken(endAt = charIndex, isSpace = isSpacingChar)
      }

      // sequence boundary follows
      1 -> if (isSpacingChar || !this.isMiddleOfWord(char, nextChar)) {
        this.addToken(endAt = charIndex, isSpace = isSpacingChar)
        this.addSentence(endAt = charIndex)
      }

      // no boundary follows
      2 -> if (isSpacingChar) {
        this.addToken(endAt = charIndex, isSpace = true)
      }
    }
  }

  /**
   * @param char a char of the text
   * @param nextChar the char that follows the given [char]
   *
   * @return a Boolean indicating if the given [char] is in the middle of a word
   */
  private fun isMiddleOfWord(char: Char, nextChar: Char): Boolean
    = !this@NeuralTokenizer.useScriptioContinua && nextChar.isLetterOrDigit() && char.isLetterOrDigit()

  /**
   * Add the given [char] to the token and sentence buffers.
   *
   * @param char the char to add
   */
  private fun addToBuffers(char: Char) {
    this.curTokenBuffer.append(char)
    this.curSentenceBuffer.append(char)
  }

  /**
   * Add a new [Token] to the list of tokens of the current sentence.
   *
   * @param endAt the index of the last character of the token
   * @param isSpace a Boolean indicating if the token is composed by a single spacing character
   */
  private fun addToken(endAt: Int, isSpace: Boolean) {

    val (id, startAt) = this.getNextTokenIdAndStart()

    this.curSentenceTokens.add(Token(
      id = id,
      form = this.curTokenBuffer.toString(),
      startAt = startAt,
      endAt = endAt,
      isSpace = isSpace && startAt == endAt
    ))

    this.resetCurTokenBuffer()
  }

  /**
   * @return a Pair containing the ID and the start index of the next token
   */
  private fun getNextTokenIdAndStart(): Pair<Int, Int> {

    val id: Int
    val startAt: Int

    if (this.curSentenceTokens.size == 0) {
      id = 0
      startAt = if (this.sentences.size > 0)
        this.sentences.last().endAt + 1
      else
        0

    } else {
      val lastToken: Token = this.curSentenceTokens.last()
      startAt = lastToken.endAt + 1
      id = lastToken.id + 1
    }

    return Pair(id, startAt)
  }

  /**
   * Add a new [Sentence] to [sentences].
   *
   * @param endAt the index of the last character of the sentence
   */
  private fun addSentence(endAt: Int) {

    val id: Int = if (this.sentences.size == 0) 0 else this.sentences.last().id + 1
    val startAt: Int = if (this.sentences.size == 0) 0 else this.sentences.last().endAt + 1

    this.sentences.add(Sentence(
      id = id,
      text = this.curSentenceBuffer.toString(),
      startAt = startAt,
      endAt = endAt,
      tokens = this.curSentenceTokens
    ))

    this.resetCurSentenceBuffer()
  }

  /**
   * Reset the currently buffered token.
   */
  private fun resetCurTokenBuffer() {
    this.curTokenBuffer.setLength(0)
  }

  /**
   * Reset the currently buffered sentence.
   */
  private fun resetCurSentenceBuffer() {
    this.curSentenceBuffer.setLength(0)
    this.curSentenceTokens = ArrayList()
  }
}

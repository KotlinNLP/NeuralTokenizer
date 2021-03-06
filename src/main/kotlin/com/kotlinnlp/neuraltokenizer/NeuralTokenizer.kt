/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.neuraltokenizer.utils.AbbreviationsContainer
import com.kotlinnlp.neuraltokenizer.utils.abbreviations
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * Neural Tokenizer.
 *
 * @property model the model for the sub-networks of this [NeuralTokenizer]
 * @param charsDropout the probability of dropout of the chars encoding (default 0.0)
 * @param boundariesDropout the probability of dropout of the boundaries classification (default 0.0)
 */
class NeuralTokenizer(
  val model: NeuralTokenizerModel,
  charsDropout: Double = 0.0,
  boundariesDropout: Double = 0.0
) {

  /**
   * The encoder of the characters of a segment.
   */
  val charsEncoder = BiRNNEncoder<DenseNDArray>(
    network = this.model.biRNN,
    dropout = charsDropout,
    propagateToInput = true)

  /**
   * The boundaries classifier.
   */
  val boundariesClassifier = BatchFeedforwardProcessor<DenseNDArray>(
    model = this.model.boundariesNetworkModel,
    dropout = boundariesDropout,
    propagateToInput = true)

  /**
   * A Boolean indicating if the language uses the "scriptio continua" style (writing without spaces).
   */
  private val useScriptioContinua: Boolean = this.model.language in setOf(
    Language.Chinese,
    Language.Japanese,
    Language.Thai)

  /**
   * The sentences resulting from the tokenization of a text.
   */
  private val sentences = mutableListOf<Sentence>()

  /**
   * The currently buffered token.
   */
  private var curTokenBuffer = StringBuffer()

  /**
   * The list of completed tokens of the currently buffered sentence.
   */
  private val curSentenceTokens = mutableListOf<Token>()

  /**
   * The number of spacing chars skipped after the last token added.
   * It is used to calculate correctly the start char index of the currently buffered token.
   */
  private var skippedSpacingChars = 0

  /**
   * Tokenize the text splitting it in [Sentence]s and [Token]s.
   *
   * @param text the text to tokenize
   *
   * @return the list of sentences which compose the [text], each containing the list of tokens
   */
  fun tokenize(text: String): List<Sentence> {

    this.initializeTokenization()

    this.forEachSegment(text) {
      this.processSegment(text = text, range = it)
    }

    return this.sentences.toList()
  }

  /**
   * @param text the whole text to tokenize
   * @param start the start index of the focus segment
   * @param length the length of the focus segment
   *
   * @return a list with the classification array of each character
   *         (0 = token boundary follows, 1 = sentence boundary follows, 2 = no boundary follows)
   */
  fun classifyChars(text: String, start: Int, length: Int): List<DenseNDArray> =
    this.boundariesClassifier.forward(
      this.charsEncoder.forward(this.charsToEmbeddings(text = text, start = start, length = length)))

  /**
   * Initialize variables used during the tokenization.
   */
  private fun initializeTokenization() {

    this.sentences.clear()
    this.skippedSpacingChars = 0
  }

  /**
   * Iterate over the text segments.
   *
   * @param text the text to tokenize
   * @param callback a callback called for each segment (it takes the range of segment indices as argument)
   */
  private fun forEachSegment(text: String, callback: (IntRange) -> Unit) {

    val tokenizer = this@NeuralTokenizer
    var startIndex = 0

    while (startIndex < text.length) {

      val segmentRange = IntRange(
        start = startIndex,
        endInclusive = minOf(startIndex + tokenizer.model.maxSegmentSize, text.length) - 1
      )

      callback(segmentRange)

      startIndex = tokenizer.getLastTokenEndIndex() + this.skippedSpacingChars + tokenizer.curTokenBuffer.length + 1
    }
  }

  /**
   * @return the end index of the last token added to the current buffer
   */
  private fun getLastTokenEndIndex(): Int = when {
    this.curSentenceTokens.size > 0 -> this.curSentenceTokens.last().position.end // new tokens added
    this.sentences.size > 0 -> this@NeuralTokenizer.sentences.last().position.end // new sentences added
    else -> 0 // first token, no new tokens or sentences added (no boundaries found)
  }

  /**
   * Process the segment of [text] within a given [range].
   *
   * @param text the text to tokenize
   * @param range the range of char indices of the segment
   */
  private fun processSegment(text: String, range: IntRange) {

    val charsClassification = this.classifyChars(text = text, start = range.first, length = range.count())
    val prevSentencesCount: Int = this.sentences.size
    val sentencePrevTokensCount: Int = this.curSentenceTokens.size

    charsClassification.forEachIndexed { i, charClassification ->

      val textIndex: Int = range.first + i

      this.processChar(
        char = text[textIndex],
        nextChar = if (textIndex < text.lastIndex) text[textIndex + 1] else null,
        charIndex = textIndex,
        charClass = charClassification.argMaxIndex())
    }

    this.shiftBuffer(prevSentencesCount = prevSentencesCount, sentencePrevTokensCount = sentencePrevTokensCount)
  }

  /**
   * Shift buffers to left based on the current prediction.
   * If new sentences are been added, buffers are shifted removing all the completed sentences.
   * If only tokens are been added, buffers are shifted removing the first N tokens until the one that crosses the
   * middle of the segment.
   * If neither sentences or tokens are added, buffers are shifted of a number of chars equal to half of the max
   * segment size (defined in the model).
   *
   * @param prevSentencesCount the number of completed sentences before processing the current segment
   * @param sentencePrevTokensCount the number of completed tokens of the current sentence before processing the current
   *                                segment
   */
  private fun shiftBuffer(prevSentencesCount: Int, sentencePrevTokensCount: Int) {

    val segmentStartsWithSpaces: Boolean =
      this.skippedSpacingChars > 0 && this.curTokenBuffer.length < this.model.maxSegmentSize

    when {
      this.sentences.size > prevSentencesCount -> // New sentences added
        this.shiftBufferBySentences()
      this.curSentenceTokens.size > sentencePrevTokensCount -> // New tokens added
        this.shiftBufferByTokens(sentencePrevTokensCount = sentencePrevTokensCount)
      segmentStartsWithSpaces -> // Shift all initial spaces
        this.curTokenBuffer.setLength(0)
      else -> // No boundaries found
        this.shiftHalfBuffer()
    }
  }

  /**
   * Shift the token buffer of an amount of chars equal to half segment.
   */
  private fun shiftHalfBuffer() {

    val halfSegmentSize: Int = this.model.maxSegmentSize / 2

    this.curTokenBuffer.delete(this.curTokenBuffer.length - halfSegmentSize, this.curTokenBuffer.length)
  }

  /**
   * Shift the sentence buffer of an amount equal to the first completed tokens until the one in the middle of the
   * current segment.
   *
   * @param sentencePrevTokensCount the number of completed tokens of the current sentence before processing the current
   *                                segment
   */
  private fun shiftBufferByTokens(sentencePrevTokensCount: Int) {

    val curSegmentTokens: List<Token> =
      this.curSentenceTokens.subList(sentencePrevTokensCount, this.curSentenceTokens.size)

    val tokensToKeep: Int = sentencePrevTokensCount + this.getSegmentTokensToKeep(curSegmentTokens)

    (tokensToKeep until this.curSentenceTokens.size).reversed().forEach { i -> this.curSentenceTokens.removeAt(i) }

    this.resetCurTokenBuffer()
  }

  /**
   * @param segment a segment of tokens
   *
   * @return the number of segment tokens to keep, following the rule of the [shiftBufferByTokens] method
   */
  private fun getSegmentTokensToKeep(segment: Iterable<Token>): Int {

    val tokensIterator = segment.iterator()
    var lastTokenEnd = segment.first().position.start - 1
    var tokensCharsCount = 0
    var tokensToKeep = 0

    while (tokensIterator.hasNext() && tokensCharsCount < this.model.maxSegmentSize / 2) {

      val token: Token = tokensIterator.next()

      tokensCharsCount += token.position.end - lastTokenEnd
      lastTokenEnd = token.position.end

      tokensToKeep++
    }

    return tokensToKeep
  }

  /**
   * Shift buffers of an amount equal to all completed sentences (= reset buffers currently not completed).
   */
  private fun shiftBufferBySentences() {

    this.resetCurSentenceBuffer()
    this.resetCurTokenBuffer()
  }

  /**
   * Associate an embeddings vector to each character of the sentence.
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

    val langCode: String = this@NeuralTokenizer.model.language.isoCode

    if (this[focusIndex] == '.' && focusIndex > 0 && langCode in abbreviations) {

      val langAbbreviations: AbbreviationsContainer = abbreviations.getValue(langCode)

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
    return this@NeuralTokenizer.model.embeddings[this].values
  }

  /**
   * Process the [char] understanding if a token or a sentence is just ended at the given [charIndex].
   *
   * @param char the char to process
   * @param nextChar the char following the [char] or null if it is the last
   * @param charIndex the index of the [char] within the text
   * @param charClass the predicted class of the [char]
   */
  private fun processChar(char: Char, nextChar: Char?, charIndex: Int, charClass: Int) {

    val isSpacingChar: Boolean = char.isWhitespace()

    if (isSpacingChar && this.curTokenBuffer.isNotEmpty())
      this.addToken(end = charIndex - 1) // automatically add the previously buffered token

    if (isSpacingChar) {

      this.skippedSpacingChars++

    } else {

      this.addToBuffer(char)

      if (nextChar == null) { // End of text
        this.addToken(end = charIndex)

      } else if (!this.isMiddleOfWord(char, nextChar)) when (charClass) {

        0 -> this.addToken(end = charIndex) // token boundary follows

        1 -> { // sentence boundary follows
          this.addToken(end = charIndex)
          this.addSentence(endAt = charIndex)
        }
      }
    }

    if (nextChar == null) this.addSentence(endAt = charIndex) // End of text
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
   * Add the given [char] to the token buffer.
   *
   * @param char the char to add
   */
  private fun addToBuffer(char: Char) {
    this.curTokenBuffer.append(char)
  }

  /**
   * Add a new [Token] to the list of tokens of the current sentence.
   *
   * @param end the index of the last character of the token
   */
  private fun addToken(end: Int) {

    val (index, start) = this.getNextTokenIndexAndStart()

    this.curSentenceTokens.add(Token(
      form = this.curTokenBuffer.toString(),
      position = Position(index = index, start = start, end = end)
    ))

    this.resetCurTokenBuffer()
  }

  /**
   * @return a Pair containing the index and the start char index of the next token
   */
  private fun getNextTokenIndexAndStart(): Pair<Int, Int> {

    val index: Int
    val start: Int

    if (this.curSentenceTokens.size == 0) {
      // don't reset the index every sentence
      index = if (this.sentences.isEmpty()) 0 else this.sentences.last().tokens.size
      start = this.getFirstTokenStart()

    } else {
      val lastToken: Token = this.curSentenceTokens.last()
      start = lastToken.position.end + this.skippedSpacingChars + 1
      index = lastToken.position.index + 1
    }

    return Pair(index, start)
  }

  /**
   * @return the start index of the first token of the current sentence
   */
  private fun getFirstTokenStart(): Int =
    this.skippedSpacingChars + if (this.sentences.isNotEmpty()) (this.sentences.last().position.end + 1) else 0

  /**
   * Add a new [Sentence] to [sentences].
   *
   * @param endAt the index of the last character of the sentence
   */
  private fun addSentence(endAt: Int) {

    val index: Int = this.sentences.size
    val startAt: Int = if (this.sentences.size == 0) 0 else this.sentences.last().position.end + 1

    this.sentences.add(Sentence(
      position = Position(index = index, start = startAt, end = endAt),
      tokens = this.curSentenceTokens.toList() // it must be a copy
    ))

    this.resetCurSentenceBuffer()
  }

  /**
   * Reset the currently buffered token.
   */
  private fun resetCurTokenBuffer() {
    this.curTokenBuffer.setLength(0)
    this.skippedSpacingChars = 0
  }

  /**
   * Reset the currently buffered sentence.
   */
  private fun resetCurSentenceBuffer() {
    this.curSentenceTokens.clear()
  }
}

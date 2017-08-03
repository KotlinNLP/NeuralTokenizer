/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.conllio.CoNLLUEvaluator
import com.kotlinnlp.conllio.CoNLLWriter
import com.kotlinnlp.neuraltokenizer.*
import java.io.File
import java.io.FileInputStream

/**
 * Execute an evaluation of a [NeuralTokenizer] over the test set read from the file given as first argument.
 * The model of the tokenizer is read from the serialized file given as second argument.
 */
fun main(args: Array<String>) {

  val OUTPUT_FILENAME = "/tmp/tokenizer_output_corpus.conll"
  val TEST_FILENAME = "/tmp/tokenizer_test_corpus.conll"

  val testSet = readDataset(args[0])

  val tokenizer = buildTokenizer(modelFilename = args[1])
  val outputSentences = tokenizer.tokenize(text = testSet.first.joinToString(""))

  CoNLLWriter.toFile(
    sentences = outputSentences.toCoNLLSentences(),
    outputFilePath = OUTPUT_FILENAME,
    writeComments = false)

  CoNLLWriter.toFile(
    sentences = buildSentences(testSet).toCoNLLSentences(),
    outputFilePath = TEST_FILENAME,
    writeComments = false)

  println(CoNLLUEvaluator.evaluate(systemFilePath = OUTPUT_FILENAME, goldFilePath = TEST_FILENAME))
}

/**
 *
 */
private fun buildTokenizer(modelFilename: String) = NeuralTokenizer(
  maxSegmentSize = 50,
  charEmbeddingsSize = 30,
  model = NeuralTokenizerModel.load(FileInputStream(File(modelFilename))))

/**
 *
 */
private fun buildSentences(dataset: Pair<ArrayList<String>, ArrayList<ArrayList<Int>>>): ArrayList<Sentence> {

  val sentences = ArrayList<Sentence>()
  val textLen: Int = 0

  dataset.first.zip(dataset.second).forEachIndexed { sentenceId, (sentence, goldClassification) ->

    sentences.add(Sentence(
      id = sentenceId,
      text = sentence,
      startAt = textLen,
      endAt = textLen + sentence.length,
      tokens = buildTokens(sentence = sentence, goldClassification = goldClassification)
    ))
  }

  return sentences
}

/**
 *
 */
private fun buildTokens(sentence: String, goldClassification: ArrayList<Int>): ArrayList<Token> {

  val tokens = ArrayList<Token>()
  var startIndex: Int = 0

  goldClassification.forEachIndexed { i, charClass ->

    if (charClass != 2) { // end of token or end of sentence
      val word: String = sentence.substring(startIndex, i + 1)

      tokens.add(Token(
        id = tokens.size,
        form = word,
        startAt = startIndex,
        endAt = i,
        isSpace = word.length == 1 && word.toCharArray()[0].isSpace()))

      startIndex = i + 1
    }
  }

  return tokens
}

/**
 * @return a Boolean indicating if this Char is a spacing character.
 */
private fun Char.isSpace(): Boolean = Regex("\\s|\\t|\\n|\\r").containsMatchIn(this.toString())

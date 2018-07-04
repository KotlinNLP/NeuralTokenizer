/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * Data class containing the properties of a sentence.
 *
 * @property tokens the list of tokens that compose the sentence
 * @property position the position of this sentence in the original text
 */
data class Sentence(
  override val tokens: List<Token>,
  override val position: Position
) : RealSentence<Token> {

  companion object {

    private const val TOKENS_SEPARATOR = " "
  }

  /**
   * The text of this sentence
   */
  val text: String by lazy {

    val text = StringBuffer()

    this.tokens.forEach {
      text.append(TOKENS_SEPARATOR.repeat(it.position.start - text.length))
      text.append(it.form)
    }

   text.toString()
  }
}

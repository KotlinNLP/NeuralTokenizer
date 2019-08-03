/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.helpers

import com.kotlinnlp.simplednn.helpers.Statistics
import com.kotlinnlp.utils.stats.MetricCounter

/**
 * Statistics given by the CoNLL evaluation script.
 *
 * @property tokens the evaluation metric of tokens
 * @property sentences the evaluation metric of sentences
 */
data class EvaluationStats(val tokens: MetricCounter, val sentences: MetricCounter) : Statistics() {

  /**
   * Reset the metrics.
   */
  override fun reset() {
    this.tokens.reset()
    this.sentences.reset()
  }

  /**
   * @return this statistics formatted into a string
   */
  override fun toString(): String = """
    - Overall accuracy : %.2f %%
    - Tokens           : $tokens
    - Sentences        : $sentences
    """
    .removePrefix("\n")
    .trimIndent()
    .format(100.0 * this.accuracy)
}

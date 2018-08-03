# NeuralTokenizer [![GitHub version](https://badge.fury.io/gh/KotlinNLP%2FNeuralTokenizer.svg)](https://badge.fury.io/gh/KotlinNLP%2FNeuralTokenizer) [![Build Status](https://travis-ci.org/KotlinNLP/NeuralTokenizer.svg?branch=master)](https://travis-ci.org/KotlinNLP/NeuralTokenizer)

NeuralTokenizer is a very simple to use text tokenizer which uses neural networks from the [SimpleDNN](https://github.com/KotlinNLP/SimpleDNN "SimpleDNN") library.

NeuralTokenizer is part of [KotlinNLP](http://kotlinnlp.com/ "KotlinNLP").


## Getting Started

### Import with Maven

```xml
<dependency>
    <groupId>com.kotlinnlp</groupId>
    <artifactId>neuraltokenizer</artifactId>
    <version>0.4.0</version>
</dependency>
```

### Examples

Try some examples of usage of NeuralTokenizer running the files in the `examples` folder.

To run the examples you need datasets of test and training that you can find
[here](https://www.dropbox.com/ "NeuralTokenizer examples datasets")

### Model Serialization

The neural model is all contained into a single class which provides simple dump() and load() methods to serialize it and afterwards load it.


## License

This software is released under the terms of the 
[Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/ "Mozilla Public License, v. 2.0")


## Contributions

We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull 
request through the [github page](https://github.com/KotlinNLP/NeuralTokenizer "NeuralTokenizer on GitHub").

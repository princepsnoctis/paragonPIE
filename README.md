# ParagonPIE (formerly ParagonNLP2)

## What is it?
ParagonPIE (ProductInformationExtraction) is a transformer based neural network designed to extract product information from lines of polish receipts.

---

## Purpose
This project is self-sufficient, although it builds up on previous two [ParagonOCR](https://github.com/princepsnoctis/paragonOCR) and [ParagonNER](https://github.com/princepsnoctis/paragonNLP) (formerly ParagonNLP).

The intended pipeline is:

```mermaid
flowchart LR
    A([Image]) --> B[[ParagonOCR]]
    B --> C([List of lines])
    C --> D[[ParagonNER]]
    D --> E([Line classifications])
    E --> F[[Line filtering algorithm]]
    F --> G([Product lines])
    G --> H[[ParagonPIE]]
    H --> I([List of product information])

```

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPDfNWopT+N1csbFX3/+AEa",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/smonga612/API/blob/main/Wine_Quality_Prediction.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#importing the dependencies"
      ],
      "metadata": {
        "id": "YK6avDCrm3MF"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.model_selection  import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "import seaborn as sns\n",
        "from sklearn.svm import SVC\n"
      ],
      "metadata": {
        "id": "fbJg8s0VnNzD"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "UEWlX6H4ZKwb"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Data Collection and Data Processing of White Wine Dataset"
      ],
      "metadata": {
        "id": "71qdVe7Fn2Qo"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data=pd.read_csv(\"/content/winequality-white.csv\",header=0)"
      ],
      "metadata": {
        "id": "9Zt3UdQnn0kk"
      },
      "execution_count": 41,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data"
      ],
      "metadata": {
        "id": "oNN6toVGoE4e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 441
        },
        "outputId": "2b39ca68-e10d-4ff8-fa97-7913cf1ffa44"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
              "0               7.0              0.27         0.36            20.7      0.045   \n",
              "1               6.3              0.30         0.34             1.6      0.049   \n",
              "2               8.1              0.28         0.40             6.9      0.050   \n",
              "3               7.2              0.23         0.32             8.5      0.058   \n",
              "4               7.2              0.23         0.32             8.5      0.058   \n",
              "...             ...               ...          ...             ...        ...   \n",
              "4843            5.6              0.34         0.25             2.5      0.046   \n",
              "4844            4.8              0.29         0.23             1.1      0.044   \n",
              "4845            6.6              0.38         0.29             2.4      0.136   \n",
              "4846            5.1              0.30         0.30             2.3      0.048   \n",
              "4847            4.4              0.54         0.09             5.1      0.038   \n",
              "\n",
              "      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
              "0                    45.0                 170.0  1.00100  3.00       0.45   \n",
              "1                    14.0                 132.0  0.99400  3.30       0.49   \n",
              "2                    30.0                  97.0  0.99510  3.26       0.44   \n",
              "3                    47.0                 186.0  0.99560  3.19       0.40   \n",
              "4                    47.0                 186.0  0.99560  3.19       0.40   \n",
              "...                   ...                   ...      ...   ...        ...   \n",
              "4843                 47.0                 182.0  0.99093  3.21       0.40   \n",
              "4844                 38.0                 180.0  0.98924  3.28       0.34   \n",
              "4845                 15.0                  93.0  0.99336  3.18       0.60   \n",
              "4846                 40.0                 150.0  0.98944  3.29       0.46   \n",
              "4847                 52.0                  97.0  0.99022  3.41       0.40   \n",
              "\n",
              "      alcohol  quality  \n",
              "0         8.8        6  \n",
              "1         9.5        6  \n",
              "2        10.1        6  \n",
              "3         9.9        6  \n",
              "4         9.9        6  \n",
              "...       ...      ...  \n",
              "4843     11.3        5  \n",
              "4844     11.9        6  \n",
              "4845      9.5        5  \n",
              "4846     12.2        6  \n",
              "4847     12.2        7  \n",
              "\n",
              "[4848 rows x 12 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9e007020-0d27-4c7e-bfb1-9008c763ff81\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>fixed acidity</th>\n",
              "      <th>volatile acidity</th>\n",
              "      <th>citric acid</th>\n",
              "      <th>residual sugar</th>\n",
              "      <th>chlorides</th>\n",
              "      <th>free sulfur dioxide</th>\n",
              "      <th>total sulfur dioxide</th>\n",
              "      <th>density</th>\n",
              "      <th>pH</th>\n",
              "      <th>sulphates</th>\n",
              "      <th>alcohol</th>\n",
              "      <th>quality</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>7.0</td>\n",
              "      <td>0.27</td>\n",
              "      <td>0.36</td>\n",
              "      <td>20.7</td>\n",
              "      <td>0.045</td>\n",
              "      <td>45.0</td>\n",
              "      <td>170.0</td>\n",
              "      <td>1.00100</td>\n",
              "      <td>3.00</td>\n",
              "      <td>0.45</td>\n",
              "      <td>8.8</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>6.3</td>\n",
              "      <td>0.30</td>\n",
              "      <td>0.34</td>\n",
              "      <td>1.6</td>\n",
              "      <td>0.049</td>\n",
              "      <td>14.0</td>\n",
              "      <td>132.0</td>\n",
              "      <td>0.99400</td>\n",
              "      <td>3.30</td>\n",
              "      <td>0.49</td>\n",
              "      <td>9.5</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>8.1</td>\n",
              "      <td>0.28</td>\n",
              "      <td>0.40</td>\n",
              "      <td>6.9</td>\n",
              "      <td>0.050</td>\n",
              "      <td>30.0</td>\n",
              "      <td>97.0</td>\n",
              "      <td>0.99510</td>\n",
              "      <td>3.26</td>\n",
              "      <td>0.44</td>\n",
              "      <td>10.1</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>7.2</td>\n",
              "      <td>0.23</td>\n",
              "      <td>0.32</td>\n",
              "      <td>8.5</td>\n",
              "      <td>0.058</td>\n",
              "      <td>47.0</td>\n",
              "      <td>186.0</td>\n",
              "      <td>0.99560</td>\n",
              "      <td>3.19</td>\n",
              "      <td>0.40</td>\n",
              "      <td>9.9</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>7.2</td>\n",
              "      <td>0.23</td>\n",
              "      <td>0.32</td>\n",
              "      <td>8.5</td>\n",
              "      <td>0.058</td>\n",
              "      <td>47.0</td>\n",
              "      <td>186.0</td>\n",
              "      <td>0.99560</td>\n",
              "      <td>3.19</td>\n",
              "      <td>0.40</td>\n",
              "      <td>9.9</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4843</th>\n",
              "      <td>5.6</td>\n",
              "      <td>0.34</td>\n",
              "      <td>0.25</td>\n",
              "      <td>2.5</td>\n",
              "      <td>0.046</td>\n",
              "      <td>47.0</td>\n",
              "      <td>182.0</td>\n",
              "      <td>0.99093</td>\n",
              "      <td>3.21</td>\n",
              "      <td>0.40</td>\n",
              "      <td>11.3</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4844</th>\n",
              "      <td>4.8</td>\n",
              "      <td>0.29</td>\n",
              "      <td>0.23</td>\n",
              "      <td>1.1</td>\n",
              "      <td>0.044</td>\n",
              "      <td>38.0</td>\n",
              "      <td>180.0</td>\n",
              "      <td>0.98924</td>\n",
              "      <td>3.28</td>\n",
              "      <td>0.34</td>\n",
              "      <td>11.9</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4845</th>\n",
              "      <td>6.6</td>\n",
              "      <td>0.38</td>\n",
              "      <td>0.29</td>\n",
              "      <td>2.4</td>\n",
              "      <td>0.136</td>\n",
              "      <td>15.0</td>\n",
              "      <td>93.0</td>\n",
              "      <td>0.99336</td>\n",
              "      <td>3.18</td>\n",
              "      <td>0.60</td>\n",
              "      <td>9.5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4846</th>\n",
              "      <td>5.1</td>\n",
              "      <td>0.30</td>\n",
              "      <td>0.30</td>\n",
              "      <td>2.3</td>\n",
              "      <td>0.048</td>\n",
              "      <td>40.0</td>\n",
              "      <td>150.0</td>\n",
              "      <td>0.98944</td>\n",
              "      <td>3.29</td>\n",
              "      <td>0.46</td>\n",
              "      <td>12.2</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4847</th>\n",
              "      <td>4.4</td>\n",
              "      <td>0.54</td>\n",
              "      <td>0.09</td>\n",
              "      <td>5.1</td>\n",
              "      <td>0.038</td>\n",
              "      <td>52.0</td>\n",
              "      <td>97.0</td>\n",
              "      <td>0.99022</td>\n",
              "      <td>3.41</td>\n",
              "      <td>0.40</td>\n",
              "      <td>12.2</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>4848 rows × 12 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9e007020-0d27-4c7e-bfb1-9008c763ff81')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-9e007020-0d27-4c7e-bfb1-9008c763ff81 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-9e007020-0d27-4c7e-bfb1-9008c763ff81');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-326e6b81-e2f7-485e-9c56-8da6555d9f79\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-326e6b81-e2f7-485e-9c56-8da6555d9f79')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-326e6b81-e2f7-485e-9c56-8da6555d9f79 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "  <div id=\"id_8d99c691-5bb1-4452-adc0-d3b245d0bd3f\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('wine_data')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_8d99c691-5bb1-4452-adc0-d3b245d0bd3f button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('wine_data');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data['volatile acidity'].value_counts()"
      ],
      "metadata": {
        "id": "kIG5UXKdUYG9",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "64de4545-b220-40a4-af19-d0f4fa0fcdec"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.280    263\n",
              "0.240    251\n",
              "0.260    240\n",
              "0.250    230\n",
              "0.220    226\n",
              "        ... \n",
              "0.850      1\n",
              "0.930      1\n",
              "0.545      1\n",
              "0.910      1\n",
              "0.405      1\n",
              "Name: volatile acidity, Length: 124, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zFC68dYUcC7v",
        "outputId": "60c4dc23-1bcc-4416-f4a2-9c3c2fa19bf4"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "fixed acidity           0\n",
              "volatile acidity        0\n",
              "citric acid             0\n",
              "residual sugar          0\n",
              "chlorides               0\n",
              "free sulfur dioxide     0\n",
              "total sulfur dioxide    0\n",
              "density                 0\n",
              "pH                      0\n",
              "sulphates               0\n",
              "alcohol                 0\n",
              "quality                 0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Y=wine_data['quality'].apply( lambda yvalue: 1 if yvalue > 5 else 0)"
      ],
      "metadata": {
        "id": "BLsFxGiemHW2"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data.loc[wine_data['quality'] <= 5, 'quality'] = 0"
      ],
      "metadata": {
        "id": "kHbmZKplPpyN"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data.loc[wine_data['quality'] > 5, 'quality'] = 1"
      ],
      "metadata": {
        "id": "WjZs9hDtSBDS"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data['quality']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gLcF_eODR6g1",
        "outputId": "c0ff731f-2b23-4db0-c40c-86da2bc94d73"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0       1\n",
              "1       1\n",
              "2       1\n",
              "3       1\n",
              "4       1\n",
              "       ..\n",
              "4843    0\n",
              "4844    1\n",
              "4845    0\n",
              "4846    1\n",
              "4847    1\n",
              "Name: quality, Length: 4848, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data.describe()"
      ],
      "metadata": {
        "id": "NB4ey64goQUb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 355
        },
        "outputId": "5027bd97-1ac3-48d0-b747-937572b3ecab"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       fixed acidity  volatile acidity  citric acid  residual sugar  \\\n",
              "count    4848.000000       4848.000000  4848.000000     4848.000000   \n",
              "mean        6.862222          0.277809     0.334398        6.409004   \n",
              "std         0.842741          0.100723     0.121233        5.079542   \n",
              "min         3.800000          0.080000     0.000000        0.600000   \n",
              "25%         6.300000          0.210000     0.270000        1.700000   \n",
              "50%         6.800000          0.260000     0.320000        5.200000   \n",
              "75%         7.300000          0.320000     0.390000        9.900000   \n",
              "max        14.200000          1.100000     1.660000       65.800000   \n",
              "\n",
              "         chlorides  free sulfur dioxide  total sulfur dioxide      density  \\\n",
              "count  4848.000000          4848.000000           4848.000000  4848.000000   \n",
              "mean      0.045830            35.321782            138.515780     0.994044   \n",
              "std       0.021925            16.992805             42.552774     0.002992   \n",
              "min       0.009000             2.000000              9.000000     0.987110   \n",
              "25%       0.036000            23.000000            108.000000     0.991740   \n",
              "50%       0.043000            34.000000            134.000000     0.993800   \n",
              "75%       0.050000            46.000000            167.000000     0.996128   \n",
              "max       0.346000           289.000000            440.000000     1.038980   \n",
              "\n",
              "                pH    sulphates      alcohol      quality  \n",
              "count  4848.000000  4848.000000  4848.000000  4848.000000  \n",
              "mean      3.188383     0.489736    10.512039     0.664604  \n",
              "std       0.151244     0.114007     1.230154     0.472177  \n",
              "min       2.720000     0.220000     8.000000     0.000000  \n",
              "25%       3.090000     0.410000     9.500000     0.000000  \n",
              "50%       3.180000     0.470000    10.400000     1.000000  \n",
              "75%       3.280000     0.550000    11.400000     1.000000  \n",
              "max       3.820000     1.060000    14.200000     1.000000  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d24041d1-4604-40cc-a45c-6717c399b68b\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>fixed acidity</th>\n",
              "      <th>volatile acidity</th>\n",
              "      <th>citric acid</th>\n",
              "      <th>residual sugar</th>\n",
              "      <th>chlorides</th>\n",
              "      <th>free sulfur dioxide</th>\n",
              "      <th>total sulfur dioxide</th>\n",
              "      <th>density</th>\n",
              "      <th>pH</th>\n",
              "      <th>sulphates</th>\n",
              "      <th>alcohol</th>\n",
              "      <th>quality</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "      <td>4848.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>6.862222</td>\n",
              "      <td>0.277809</td>\n",
              "      <td>0.334398</td>\n",
              "      <td>6.409004</td>\n",
              "      <td>0.045830</td>\n",
              "      <td>35.321782</td>\n",
              "      <td>138.515780</td>\n",
              "      <td>0.994044</td>\n",
              "      <td>3.188383</td>\n",
              "      <td>0.489736</td>\n",
              "      <td>10.512039</td>\n",
              "      <td>0.664604</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.842741</td>\n",
              "      <td>0.100723</td>\n",
              "      <td>0.121233</td>\n",
              "      <td>5.079542</td>\n",
              "      <td>0.021925</td>\n",
              "      <td>16.992805</td>\n",
              "      <td>42.552774</td>\n",
              "      <td>0.002992</td>\n",
              "      <td>0.151244</td>\n",
              "      <td>0.114007</td>\n",
              "      <td>1.230154</td>\n",
              "      <td>0.472177</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>3.800000</td>\n",
              "      <td>0.080000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.600000</td>\n",
              "      <td>0.009000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>9.000000</td>\n",
              "      <td>0.987110</td>\n",
              "      <td>2.720000</td>\n",
              "      <td>0.220000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>6.300000</td>\n",
              "      <td>0.210000</td>\n",
              "      <td>0.270000</td>\n",
              "      <td>1.700000</td>\n",
              "      <td>0.036000</td>\n",
              "      <td>23.000000</td>\n",
              "      <td>108.000000</td>\n",
              "      <td>0.991740</td>\n",
              "      <td>3.090000</td>\n",
              "      <td>0.410000</td>\n",
              "      <td>9.500000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>6.800000</td>\n",
              "      <td>0.260000</td>\n",
              "      <td>0.320000</td>\n",
              "      <td>5.200000</td>\n",
              "      <td>0.043000</td>\n",
              "      <td>34.000000</td>\n",
              "      <td>134.000000</td>\n",
              "      <td>0.993800</td>\n",
              "      <td>3.180000</td>\n",
              "      <td>0.470000</td>\n",
              "      <td>10.400000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>7.300000</td>\n",
              "      <td>0.320000</td>\n",
              "      <td>0.390000</td>\n",
              "      <td>9.900000</td>\n",
              "      <td>0.050000</td>\n",
              "      <td>46.000000</td>\n",
              "      <td>167.000000</td>\n",
              "      <td>0.996128</td>\n",
              "      <td>3.280000</td>\n",
              "      <td>0.550000</td>\n",
              "      <td>11.400000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>14.200000</td>\n",
              "      <td>1.100000</td>\n",
              "      <td>1.660000</td>\n",
              "      <td>65.800000</td>\n",
              "      <td>0.346000</td>\n",
              "      <td>289.000000</td>\n",
              "      <td>440.000000</td>\n",
              "      <td>1.038980</td>\n",
              "      <td>3.820000</td>\n",
              "      <td>1.060000</td>\n",
              "      <td>14.200000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d24041d1-4604-40cc-a45c-6717c399b68b')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-d24041d1-4604-40cc-a45c-6717c399b68b button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-d24041d1-4604-40cc-a45c-6717c399b68b');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-5c899c36-a5fc-428b-a544-e041f4a13271\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-5c899c36-a5fc-428b-a544-e041f4a13271')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-5c899c36-a5fc-428b-a544-e041f4a13271 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data.shape"
      ],
      "metadata": {
        "id": "WOvmTpSsoZFj",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "872ebaa4-0978-45a5-c530-5edf086bc9b3"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(4848, 12)"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data['quality'].value_counts()"
      ],
      "metadata": {
        "id": "zvILcTtHot6k",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "11cec690-7de8-43a2-e5bd-72246b3b2f3d"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1    3222\n",
              "0    1626\n",
              "Name: quality, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "wine_data.groupby('quality').mean()"
      ],
      "metadata": {
        "id": "vFMgFLY7pCaD",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 161
        },
        "outputId": "2c046140-8f8b-480c-e7b6-e259fd2d9724"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         fixed acidity  volatile acidity  citric acid  residual sugar  \\\n",
              "quality                                                                 \n",
              "0             6.968204          0.309978     0.334563        7.075892   \n",
              "1             6.808737          0.261575     0.334314        6.072455   \n",
              "\n",
              "         chlorides  free sulfur dioxide  total sulfur dioxide   density  \\\n",
              "quality                                                                   \n",
              "0         0.051487            35.325646            148.695264  0.995175   \n",
              "1         0.042975            35.319832            133.378647  0.993474   \n",
              "\n",
              "               pH  sulphates    alcohol  \n",
              "quality                                  \n",
              "0        3.170234   0.481371   9.849444  \n",
              "1        3.197542   0.493957  10.846420  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8a7e5b4f-843e-47e9-b751-8ba55ea116f9\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>fixed acidity</th>\n",
              "      <th>volatile acidity</th>\n",
              "      <th>citric acid</th>\n",
              "      <th>residual sugar</th>\n",
              "      <th>chlorides</th>\n",
              "      <th>free sulfur dioxide</th>\n",
              "      <th>total sulfur dioxide</th>\n",
              "      <th>density</th>\n",
              "      <th>pH</th>\n",
              "      <th>sulphates</th>\n",
              "      <th>alcohol</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>quality</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>6.968204</td>\n",
              "      <td>0.309978</td>\n",
              "      <td>0.334563</td>\n",
              "      <td>7.075892</td>\n",
              "      <td>0.051487</td>\n",
              "      <td>35.325646</td>\n",
              "      <td>148.695264</td>\n",
              "      <td>0.995175</td>\n",
              "      <td>3.170234</td>\n",
              "      <td>0.481371</td>\n",
              "      <td>9.849444</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>6.808737</td>\n",
              "      <td>0.261575</td>\n",
              "      <td>0.334314</td>\n",
              "      <td>6.072455</td>\n",
              "      <td>0.042975</td>\n",
              "      <td>35.319832</td>\n",
              "      <td>133.378647</td>\n",
              "      <td>0.993474</td>\n",
              "      <td>3.197542</td>\n",
              "      <td>0.493957</td>\n",
              "      <td>10.846420</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8a7e5b4f-843e-47e9-b751-8ba55ea116f9')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-8a7e5b4f-843e-47e9-b751-8ba55ea116f9 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-8a7e5b4f-843e-47e9-b751-8ba55ea116f9');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-7c7483d0-de2b-4556-9a76-5e58695b951f\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-7c7483d0-de2b-4556-9a76-5e58695b951f')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-7c7483d0-de2b-4556-9a76-5e58695b951f button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "separating data and labels"
      ],
      "metadata": {
        "id": "EJV8OXBnvNdc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X=wine_data.drop(columns='quality',axis=1)"
      ],
      "metadata": {
        "id": "f2bshPCLvDGT"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X)"
      ],
      "metadata": {
        "id": "ZnuAbo5Vvg7B",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a3472f8a-cf54-4cbd-be58-5dcbcea11885"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
            "0               7.0              0.27         0.36            20.7      0.045   \n",
            "1               6.3              0.30         0.34             1.6      0.049   \n",
            "2               8.1              0.28         0.40             6.9      0.050   \n",
            "3               7.2              0.23         0.32             8.5      0.058   \n",
            "4               7.2              0.23         0.32             8.5      0.058   \n",
            "...             ...               ...          ...             ...        ...   \n",
            "4843            5.6              0.34         0.25             2.5      0.046   \n",
            "4844            4.8              0.29         0.23             1.1      0.044   \n",
            "4845            6.6              0.38         0.29             2.4      0.136   \n",
            "4846            5.1              0.30         0.30             2.3      0.048   \n",
            "4847            4.4              0.54         0.09             5.1      0.038   \n",
            "\n",
            "      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
            "0                    45.0                 170.0  1.00100  3.00       0.45   \n",
            "1                    14.0                 132.0  0.99400  3.30       0.49   \n",
            "2                    30.0                  97.0  0.99510  3.26       0.44   \n",
            "3                    47.0                 186.0  0.99560  3.19       0.40   \n",
            "4                    47.0                 186.0  0.99560  3.19       0.40   \n",
            "...                   ...                   ...      ...   ...        ...   \n",
            "4843                 47.0                 182.0  0.99093  3.21       0.40   \n",
            "4844                 38.0                 180.0  0.98924  3.28       0.34   \n",
            "4845                 15.0                  93.0  0.99336  3.18       0.60   \n",
            "4846                 40.0                 150.0  0.98944  3.29       0.46   \n",
            "4847                 52.0                  97.0  0.99022  3.41       0.40   \n",
            "\n",
            "      alcohol  \n",
            "0         8.8  \n",
            "1         9.5  \n",
            "2        10.1  \n",
            "3         9.9  \n",
            "4         9.9  \n",
            "...       ...  \n",
            "4843     11.3  \n",
            "4844     11.9  \n",
            "4845      9.5  \n",
            "4846     12.2  \n",
            "4847     12.2  \n",
            "\n",
            "[4848 rows x 11 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(X)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Gwx2nLdlUE3y",
        "outputId": "8c0c363c-b760-4c26-d638-1842149ba414"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
            "0               7.0              0.27         0.36            20.7      0.045   \n",
            "1               6.3              0.30         0.34             1.6      0.049   \n",
            "2               8.1              0.28         0.40             6.9      0.050   \n",
            "3               7.2              0.23         0.32             8.5      0.058   \n",
            "4               7.2              0.23         0.32             8.5      0.058   \n",
            "...             ...               ...          ...             ...        ...   \n",
            "4843            5.6              0.34         0.25             2.5      0.046   \n",
            "4844            4.8              0.29         0.23             1.1      0.044   \n",
            "4845            6.6              0.38         0.29             2.4      0.136   \n",
            "4846            5.1              0.30         0.30             2.3      0.048   \n",
            "4847            4.4              0.54         0.09             5.1      0.038   \n",
            "\n",
            "      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
            "0                    45.0                 170.0  1.00100  3.00       0.45   \n",
            "1                    14.0                 132.0  0.99400  3.30       0.49   \n",
            "2                    30.0                  97.0  0.99510  3.26       0.44   \n",
            "3                    47.0                 186.0  0.99560  3.19       0.40   \n",
            "4                    47.0                 186.0  0.99560  3.19       0.40   \n",
            "...                   ...                   ...      ...   ...        ...   \n",
            "4843                 47.0                 182.0  0.99093  3.21       0.40   \n",
            "4844                 38.0                 180.0  0.98924  3.28       0.34   \n",
            "4845                 15.0                  93.0  0.99336  3.18       0.60   \n",
            "4846                 40.0                 150.0  0.98944  3.29       0.46   \n",
            "4847                 52.0                  97.0  0.99022  3.41       0.40   \n",
            "\n",
            "      alcohol  \n",
            "0         8.8  \n",
            "1         9.5  \n",
            "2        10.1  \n",
            "3         9.9  \n",
            "4         9.9  \n",
            "...       ...  \n",
            "4843     11.3  \n",
            "4844     11.9  \n",
            "4845      9.5  \n",
            "4846     12.2  \n",
            "4847     12.2  \n",
            "\n",
            "[4848 rows x 11 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Y=wine_data['quality']"
      ],
      "metadata": {
        "id": "Qlu9QqSRvo75"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(Y)"
      ],
      "metadata": {
        "id": "buOzoqvevtrw",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e5d0ffc6-0bf4-4d76-adaa-e48d4f0b09bf"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0       1\n",
            "1       1\n",
            "2       1\n",
            "3       1\n",
            "4       1\n",
            "       ..\n",
            "4843    0\n",
            "4844    1\n",
            "4845    0\n",
            "4846    1\n",
            "4847    1\n",
            "Name: quality, Length: 4848, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Training and Testing Data"
      ],
      "metadata": {
        "id": "nwizmqBgv4Qt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,Y_train,Y_test=train_test_split(X.values,Y.values,test_size=0.2,stratify=Y,random_state=7)"
      ],
      "metadata": {
        "id": "m9XhBFQbv7RU"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X_train.shape,Y_train.shape)\n"
      ],
      "metadata": {
        "id": "CA8oGd03vxUQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a015bd64-9c4d-4323-d351-99b1b86cee46"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(3878, 11) (3878,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "classifier=SVC(kernel='linear')\n"
      ],
      "metadata": {
        "id": "v7ChdlunZGaC"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "classifier.fit(X_train,Y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "YgMDtjBSZV5D",
        "outputId": "b8673456-9123-4543-f7bf-bb649df9061f"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "SVC(kernel='linear')"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>SVC(kernel=&#x27;linear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SVC</label><div class=\"sk-toggleable__content\"><pre>SVC(kernel=&#x27;linear&#x27;)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ZcWFu1RHZhYk"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#accuracy on training data"
      ],
      "metadata": {
        "id": "PLUoLoqgyDwT"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NlBNRQT9Xdzi",
        "outputId": "f0ab9e25-a9d7-4704-f562-c5ddcdeb095b"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 6.    0.24  0.34 ...  3.44  0.44 11.45]\n",
            " [ 6.9   0.28  0.28 ...  3.03  0.56 10.4 ]\n",
            " [ 5.8   0.28  0.35 ...  3.28  0.5  10.2 ]\n",
            " ...\n",
            " [ 6.    0.29  0.25 ...  3.08  0.43 13.2 ]\n",
            " [ 6.5   0.08  0.33 ...  3.34  0.7  12.  ]\n",
            " [ 6.6   0.24  0.27 ...  3.24  0.51  9.2 ]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_train_prediction=classifier.predict(X_train)\n"
      ],
      "metadata": {
        "id": "VPZcIFmbycNA"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X_train_prediction)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VBbvOuQDXU7T",
        "outputId": "483fe371-68b7-46c7-be5d-b3ce35a84ab0"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1 1 1 ... 1 1 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(Y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FwDfHZD5XvzA",
        "outputId": "2dda5d6a-4329-409f-f382-5a959bb99081"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1 1 0 ... 1 1 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "training_data_accuracy=accuracy_score(X_train_prediction, Y_train)"
      ],
      "metadata": {
        "id": "XS15Vf8mXSK3"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Accuracy on training data \",training_data_accuracy)"
      ],
      "metadata": {
        "id": "iM3opvNnISru",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d576c73b-84fc-4760-dc7c-761db2f7135f"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy on training data  0.7478081485301702\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_test_prediction=classifier.predict(X_test)\n",
        "test_data_accuracy=accuracy_score(X_test_prediction, Y_test)"
      ],
      "metadata": {
        "id": "l-yHdK_mIfpa"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Accuracy on test data \",test_data_accuracy)"
      ],
      "metadata": {
        "id": "wxnfG2YNJFrx",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f0752738-f996-49bc-8c8f-56726468c1e3"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy on test data  0.7536082474226804\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "input_data=(6.2,0.53,0.02,0.9,0.035,6,81,0.99234,3.24,0.35,9.5)\n",
        "input_data_as_numpy_array=np.asarray(input_data)\n"
      ],
      "metadata": {
        "id": "bEjv-O5jJMtO"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)"
      ],
      "metadata": {
        "id": "I3BWrul2LZ2-"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(input_data_reshaped)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3yKbDGdzimXR",
        "outputId": "c7a5e0c0-0e18-4160-a6b6-0177f9174a09"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[6.2000e+00 5.3000e-01 2.0000e-02 9.0000e-01 3.5000e-02 6.0000e+00\n",
            "  8.1000e+01 9.9234e-01 3.2400e+00 3.5000e-01 9.5000e+00]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "prediction=classifier.predict(input_data_reshaped)"
      ],
      "metadata": {
        "id": "tqwiZ3nqLjDD"
      },
      "execution_count": 39,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(prediction)"
      ],
      "metadata": {
        "id": "9wxK84hFMZKa",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "cfad1b7f-0ccb-434c-a3dc-6d1ce156f0fa"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0]\n"
          ]
        }
      ]
    }
  ]
}
# Tesla Stock Price Forecasting

Welcome to the **Tesla Stock Price Forecasting** project! This project is all about predicting the future prices of Tesla's stock. Instead of focusing only on getting the most accurate predictions, we've compared different forecasting methods to see which ones work best for this task. 

<a href="https://tslastockprediction-argetlam.streamlit.app/" target="_blank">Project Overview </a>

In this project, we explored a variety of models that are commonly used in forecasting stock prices. Our main goal was to understand the strengths and weaknesses of these models, especially when used with small datasets like ours. We also tested some "mix models," which combine different approaches to try to get better results.

### Key Insights

- **Data Size Matters**: One of the biggest challenges we faced was that our dataset was small. This made it difficult for more complex models to learn effectively, which can lead to less accurate predictions.
- **Model Performance**: 
  - **Prophet Model**: Great at capturing trends and seasonal patterns but struggled with making accurate predictions.
  - **SARIMA Model**: Good for detecting seasonality but requires a lot of storage space and isn't the best for price predictions.
  - **VAR Model**: Focuses on relationships between variables like opening and closing prices. It's promising, but it needs more data to shine.
  - **LightGBM**: This model performed well even with complex data and gave solid predictions, especially when combined with other models.
  - **LSTM**: A deep learning model that's powerful for time series data. It worked best when kept simple due to our small dataset.

### The Best Combinations

We found that combining models led to better results:
- **Prophet + LightGBM**: Combining the strengths of these two models led to more accurate predictions than using LightGBM alone.
- **LSTM + LightGBM**: This combination was the most successful, delivering the best predictions in our project.

## Recommendations

Based on our findings:
- **For Small Businesses**: Use tree-based models like LightGBM. They handle various scenarios well and are cost-effective.
- **For Medium-Sized Businesses**: Consider mix models to enhance accuracy. Combining different approaches can give you an edge.
- **For Large Corporations**: Compare and combine the most successful models to achieve the highest accuracy.

## Conclusion

Our project shows that the size of the dataset, time, and budget can greatly influence the choice of forecasting models. While simple models are quick, they may not always provide the best results. On the other hand, mix models can offer better accuracy but may sometimes overfit the data. 

Thank you for checking out our project! We hope our insights help you in choosing the right forecasting approach for your needs.


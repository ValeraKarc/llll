        if not results:
            st.error("Ни одна модель не обучилась.")
            st.stop()

        # --- Выбор лучшей модели ---
        best_name = min(results, key=lambda x: results[x]['rmse'])
        best = results[best_name]

        # --- Вывод метрик в карточках ---
        st.subheader("📊 Результаты прогнозирования")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Лучшая модель", best_name)
        with col2:
            st.metric("📉 RMSE на тесте", f"{best['rmse']:.2f}")
        with col3:
            st.metric("📈 MAPE на тесте", f"{best['mape']:.2f}%")

        # --- Сравнение всех моделей в таблице ---
        st.write("**Сводка по всем моделям:**")
        summary_data = []
        for name, res in results.items():
            summary_data.append({
                'Модель': name,
                'RMSE': f"{res['rmse']:.2f}",
                'MAPE': f"{res['mape']:.2f}%"
            })
        st.dataframe(pd.DataFrame(summary_data).sort_values('RMSE'), use_container_width=True)

        # --- Выбор модели для графика (по умолчанию лучшая) ---
        selected_model = st.selectbox(
            "Выберите модель для отображения графика",
            list(results.keys()),
            index=list(results.keys()).index(best_name)
        )
        selected = results[selected_model]

        # --- Финальный прогноз для выбранной модели ---
        full_ts = pd.concat([train, test])
        offset = to_offset(freq)
        start_date = full_ts.index[-1] + offset
        future_dates = pd.date_range(start=start_date, periods=horizon, freq=freq)

        if selected_model == 'Holt-Winters':
            model_full = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                              seasonal_periods=seasonal_periods,
                                              initialization_method='estimated').fit()
            forecast = model_full.forecast(horizon)
            try:
                pred_obj = model_full.get_prediction(start=future_dates[0], end=future_dates[-1])
                summary = pred_obj.summary_frame(alpha=0.05)
                pi_lower, pi_upper = summary['pi_lower'].values, summary['pi_upper'].values
            except:
                resid_std = np.std(train - selected['pred_test'])
                pi_lower, pi_upper = forecast - 1.96*resid_std, forecast + 1.96*resid_std
        elif selected_model == 'ARIMA':
            model_full = pm.auto_arima(full_ts, seasonal=True, m=seasonal_periods,
                                       suppress_warnings=True, error_action='ignore',
                                       stepwise=True, trace=False)
            forecast, conf_int = model_full.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
            pi_lower, pi_upper = conf_int[:, 0], conf_int[:, 1]
        else:  # ML модели
            model_full = selected['model']
            forecast = recursive_forecast(model_full, full_ts, future_dates, lags, freq)
            test_pred = selected['pred_test']
            resid_std = np.std(np.array(test) - np.array(test_pred))
            pi_lower, pi_upper = forecast - 1.96*resid_std, forecast + 1.96*resid_std

        # --- Интерактивный график с зумом и выбором модели ---
        st.subheader(f"📉 Прогноз модели: **{selected_model}**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Тренировочные',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines+markers', name='Тестовые',
                                 line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Прогноз',
                                 line=dict(color='green')))
        fig.add_trace(go.Scatter(
            x=np.concatenate([future_dates, future_dates[::-1]]),
            y=np.concatenate([pi_upper, pi_lower[::-1]]),
            fill='toself', fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% доверит. интервал'
        ))
        fig.add_vline(x=test.index[0], line_dash="dash", line_color="red", annotation_text="Начало прогноза")
        fig.update_layout(title=f"Прогноз • {selected_model}",
                          xaxis_title="Дата", yaxis_title="Total",
                          hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

        # --- Кнопка PDF (работает для отображаемой модели) ---
        if st.button("📄 Скачать PDF-отчёт"):
            # ... (ваш код PDF остаётся без изменений, используйте selected_model, forecast и т.п.)

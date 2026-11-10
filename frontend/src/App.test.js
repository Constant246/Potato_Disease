import { render, screen } from '@testing-library/react';
import App from './App';

test('renders identify potato disease title', () => {
  render(<App />);
  const titleElement = screen.getByText(/Identify Potato Disease/i);
  expect(titleElement).toBeInTheDocument();
});


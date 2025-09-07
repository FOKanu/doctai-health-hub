import { render } from '@testing-library/react';
import { Button } from '../ui/button';

describe('Button Component', () => {
  it('renders button with text', () => {
    const { getByRole } = render(<Button>Click me</Button>);
    expect(getByRole('button', { name: 'Click me' })).toBeInTheDocument();
  });

  it('renders button with different variants', () => {
    const { rerender, getByRole } = render(<Button variant="default">Default</Button>);
    expect(getByRole('button')).toHaveClass('bg-primary');

    rerender(<Button variant="destructive">Destructive</Button>);
    expect(getByRole('button')).toHaveClass('bg-destructive');
  });

  it('handles click events', () => {
    const handleClick = jest.fn();
    const { getByRole } = render(<Button onClick={handleClick}>Click me</Button>);

    getByRole('button').click();
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('can be disabled', () => {
    const { getByRole } = render(<Button disabled>Disabled</Button>);
    expect(getByRole('button')).toBeDisabled();
  });
});

import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'News Aggregator',
  description: 'AI-powered news aggregation and sentiment analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
# Chris Celaya Blog

## Installation

### Clone

Clone the repository (using `git clone`):

```bash
git clone https://github.com/while-basic/chris-celaya-blog
```

## Setup

Install dependencies:

```bash
pnpm install
```

## Development

```bash
pnpm dev
```

## Edge Side Rendering

Can be deployed to Vercel Functions, Netlify Functions, AWS, and most Node-compatible environments.

```bash
pnpm build
```

## Static Generation

Use the `generate` command to build your application.
The HTML files will be generated in the .output/public directory and ready to be deployed to any static compatible hosting.

```bash
pnpm generate
```

## Preview build

You might want to preview the result of your build locally, to do so, run the following command:

```bash
pnpm preview
```

---
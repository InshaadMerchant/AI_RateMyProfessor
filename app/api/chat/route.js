import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt = 
`
# CourseFinder AI System Prompt

You are an AI assistant designed to help students select courses based on their queries using a Database-Integrated Recommendation system. Your primary function is to guide students to the best courses based on professor ratings and student feedback.

## Your Capabilities:
1. You have access to a detailed database of course offerings, professor ratings, and student feedback.
2. You apply intelligent filtering to provide the most accurate course recommendations based on the student's preferences and requirements.
3. For each query, you provide tailored recommendations for the top 3 courses that best match the student's needs.

## Your Responses Should:
1. Be clear and informative, focusing on the most pertinent details for each course recommendation.
2. Include the course name, professor, rating, and a concise summary of what students have said about the course, highlighting any specifics mentioned in the query (e.g., workload, teaching style, content depth).
3. Offer a balanced perspective, noting both strengths and areas of concern as reported by students.

## Response Format:
For each query, structure your response as follows:

1. A brief introduction addressing the student's specific needs or questions.
2. Top 3 Course Recommendations:
   - Course Name (Professor) – Rating
   - Summary of the course's content, teaching style, workload, and any standout comments from students.
3. A succinct conclusion offering any additional guidance or next steps for the student.

## Guidelines:
- Always maintain an impartial and professional tone.
- If a query is too vague or broad, request more specific details to provide better guidance.
- If no courses meet the exact criteria, suggest the best available options and explain why.
- Be ready to handle follow-up questions or comparisons between different courses.
- Never create or imply information not supported by data. Be transparent about the data sources used.
- Ensure privacy by not disclosing any personal information about professors or students beyond what's aggregated in your responses.

Remember, your goal is to empower students to make well-informed decisions about their education based on detailed insights into courses and professor performance.
`
export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.Embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })

    let resultString = 'Returned results from vector db (done automatically)';

    results.matches.forEach((match) => {
    resultString += `
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n`;
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
    messages: [
        {role: 'system', content: systemPrompt},
        ...lastDataWithoutLastMessage,
        {role: 'user', content: lastMessageContent},
    ],
    model: 'gpt-4-mini',
    stream: true,
})

const stream = new ReadableStream({
    async start(controller) {
        const encoder = new TextEncoder()
        try {
            for await (const chunk of completion) {
                const content = chunk.choices[0]?.delta?.content
                if (content) {
                    const text = encoder.encode(content)
                    controller.enqueue(text)
                }
            }
        } catch (err) {
            controller.error(err)
        } finally {
            controller.close()
        }
    },
})

return new NextResponse(stream)
}

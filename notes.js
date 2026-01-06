import axios from "axios";

export const createComment = async (req, res) => {
    const { commentText } = req.body;

    try {
        // 1. Ask the Python Service if this is toxic
        const aiResponse = await axios.post("https://my-toxicity-api.onrender.com/predict", {
            text: commentText
        });

        // 2. Check the result
        if (aiResponse.data.is_toxic) {
             return res.status(400).json({ message: "Comment rejected: Toxicity detected." });
        }

        // 3. If safe, save to MongoDB
        const newComment = await Comment.create({ text: commentText, user: req.user._id });
        res.json(newComment);

    } catch (error) {
        res.status(500).json({ message: "AI Service Down" });
    }
};

// run the app.py  - uvicorn app:app --reload

/* testing - open up 2 separate terminals one running the app.py another sending json data!
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "You are stupid and I hate you"}'

*/
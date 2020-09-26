const puppeteer = require('puppeteer');
const fs = require('fs');

let counter = 0;

async function main() {
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();
    while (true) {
        //let url = "https://www.tripadvisor.com/Hotel_Review-g187870-d233394-Reviews-or" + counter + "-Al_Ponte_Antico_Hotel-Venice_Veneto.html#REVIEWS"
        let url = "https://www.tripadvisor.de/Hotel_Review-g187291-d647452-Reviews-or" + counter + "-Movenpick_Hotel_Stuttgart_Airport-Stuttgart_Baden_Wurttemberg.html#REVIEWS"
        await page.goto(url)

        const reviews = await page.$$('._3hDPbqWO')
        for (var i = 0; i < reviews.length; i++) {
            const review = reviews[i]
            try {
                const reviewText = await page.evaluate(review => review.textContent, review);
                const writeStream = fs.createWriteStream((i + counter) + '.txt', { flags: 'a' });
                writeStream.write(reviewText)
                writeStream.end();
            } catch (err) {
                console.log(err)
            }
        }
        counter += 5
    }
}

main()
//
//  ViewController.swift
//  SpamClassifier
//
//  Created by Nikhil Kulkarni on 8/12/18.
//  Copyright © 2018 Nikhil Kulkarni. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {

    @IBOutlet weak var spamLabel: UILabel!
    @IBOutlet weak var textField: UITextField!
    
    @IBAction func classify(_ sender: Any) {
        predict(fieldText: textField.text)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    func predict(fieldText: String?) {
        guard let text = fieldText else {
            return
        }
        
        let model = SpamClassifier()
        let message = createMessage(text: text)
        
        let input = SpamClassifierInput(message: message)
        
        do {
            let output = try model.prediction(input: input)
            spamLabel.text = output.spam_or_not
        } catch {
            spamLabel.text = "Failed to classify"
        }
    }
    
    func createMessage(text: String) -> MLMultiArray {
        let wordsFile = Bundle.main.path(forResource: "wordlist", ofType: "txt")
        let smsFile = Bundle.main.path(forResource: "SMSSpamCollection", ofType: "txt")
        
        do {
            //read words file
            let wordsFileText = try String(contentsOfFile: wordsFile!, encoding: String.Encoding.utf8)
            var wordsData = wordsFileText.components(separatedBy: .newlines)
            wordsData.removeLast() // Trailing newline.
            //read spam collection file
            let smsFileText = try String(contentsOfFile: smsFile!, encoding: String.Encoding.utf8)
            var smsData = smsFileText.components(separatedBy: .newlines)
            smsData.removeLast() // Trailing newline.
            let wordsInMessage = text.split(separator: " ")
            //create a multi-dimensional array
            let vectorized = try MLMultiArray(shape: [NSNumber(integerLiteral: wordsData.count)], dataType: MLMultiArrayDataType.double)
            for i in 0..<wordsData.count{
                let word = wordsData[i]
                if text.contains(word){
                    var wordCount = 0
                    for substr in wordsInMessage{
                        if substr.elementsEqual(word){
                            wordCount += 1
                        }
                    }
                    let tf = Double(wordCount) / Double(wordsInMessage.count)
                    var docCount = 0
                    for sms in smsData{
                        if sms.contains(word) {
                            docCount += 1
                        }
                    }
                    let idf = log(Double(smsData.count) / Double(docCount))
                    vectorized[i] = NSNumber(value: tf * idf)
                } else {
                    vectorized[i] = 0.0
                }
            }
            return vectorized
        } catch {
            return MLMultiArray()
        }
    }

}

